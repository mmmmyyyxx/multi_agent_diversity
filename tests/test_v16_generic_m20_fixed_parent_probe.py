from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import inspect
import json
import shutil
import uuid
from dataclasses import asdict
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def registry():
    module = load(
        "generic_m20_registry",
        "scripts/build_v16_generic_m20_probe_registry.py",
    )
    return module.build_registry("0" * 40)


@pytest.fixture
def workspace_tmp():
    path = (
        ROOT / "runs" / f"_test_v16_generic_m20_{uuid.uuid4().hex}"
    ).resolve()
    assert ROOT.resolve() in path.parents
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        assert ROOT.resolve() in path.parents
        try:
            shutil.rmtree(path)
        except PermissionError:
            # Windows may briefly retain a SQLite WAL handle after object
            # teardown. The path is ignored, repository-local test scratch.
            pass


def test_exact_eight_case_reconstruction_and_balanced_order(registry):
    assert registry["case_count"] == 8
    assert registry["cell_count"] == 16
    assert registry["maximum_planned_candidates"] == 32
    assert [case["source_seed"] for case in registry["cases"]] == [
        48, 48, 49, 49, 50, 50, 51, 51
    ]
    for index, case in enumerate(registry["cases"]):
        expected = (
            ["g0_fixed_target_generic", "m20_current_v15"]
            if index % 2 == 0
            else ["m20_current_v15", "g0_fixed_target_generic"]
        )
        assert case["cell_order"] == expected
        actual = hashlib.sha256(
            json.dumps(
                sorted(case["assigned_question_hashes"]),
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        assert actual == case["frozen_responsibility_evidence_hash"]


def test_preflight_proves_no_g0_leakage_and_shared_parent_target_budget(
    registry, workspace_tmp
):
    module = load(
        "generic_m20_preflight",
        "scripts/preflight_v16_generic_m20_probe.py",
    )
    result = module.preflight(registry, scratch=workspace_tmp)
    assert result["status"] == "PASS"
    assert result["api_calls"] == result["model_calls"] == 0
    assert result["g0_responsibility_leakage_count"] == 0
    assert result["cell_count"] == 16
    for row in result["cases"]:
        assert row["g0_generator_responsibility_hash_count"] == 0
        assert row["g0_context_type"] == "AccuracyDiagnosisContext"
        assert row["m20_context_type"] == "SingleLaneDiagnosisContext"
        assert row["budgets"]["g0_fixed_target_generic"] == row["budgets"][
            "m20_current_v15"
        ]
        assert row["evaluation_policy"]["candidate_acceptance_policy"] == (
            "fixed_peer_monotone_target_or_vote"
        )


def test_m20_context_is_byte_current_and_g0_uses_empty_hashes(
    registry, workspace_tmp
):
    support = load(
        "generic_m20_support_context",
        "scripts/generic_m20_probe_support.py",
    )
    case = registry["cases"][0]
    frozen = set(case["assigned_question_hashes"])
    assert support.generation_hashes(support.G0, frozen) == set()
    assert support.generation_hashes(support.M20, frozen) == frozen
    probe = support.generation_system(
        case,
        support.M20,
        out_dir=workspace_tmp / "probe",
        cache_path="",
    )
    canonical = support.system_for(
        case,
        setting="shared_responsibility_conditioned_dual_target",
        out_dir=workspace_tmp / "canonical",
        cache_path="",
    )
    target = int(case["target_agent_id"])
    left, _ = probe._proposal_context(
        target, probe.agents[target].current_prompt, frozen
    )
    right, _ = canonical._proposal_context(
        target, canonical.agents[target].current_prompt, frozen
    )
    from multi_dataset_diverse_rl.tcs import build_teacher_request, serialize_context

    assert serialize_context(left) == serialize_context(right)
    assert build_teacher_request(left) == build_teacher_request(right)


def test_both_arms_use_identical_m20_common_safe_evaluator(
    registry, workspace_tmp
):
    support = load(
        "generic_m20_support_eval",
        "scripts/generic_m20_probe_support.py",
    )
    case = registry["cases"][0]
    left = support.evaluation_system(
        case,
        out_dir=workspace_tmp / "left",
        cache_path="",
    )
    right = support.evaluation_system(
        case,
        out_dir=workspace_tmp / "right",
        cache_path="",
    )
    assert left.protocol == right.protocol
    assert left.protocol.sample_pool_policy == "member_aware_residuals"
    assert left.protocol.stage_a_policy == "matched_all_generated"
    assert left.protocol.candidate_acceptance_policy == (
        "fixed_peer_monotone_target_or_vote"
    )
    assert support.state_hash(left) == support.state_hash(right)


def test_core_state_hash_detects_optimizer_mutation(registry, workspace_tmp):
    support = load(
        "generic_m20_support_state",
        "scripts/generic_m20_probe_support.py",
    )
    case = registry["cases"][0]
    system = support.evaluation_system(
        case,
        out_dir=workspace_tmp / "state",
        cache_path="",
    )
    before = support.state_hash(system)
    system.selected_target_ids.append(4)
    assert support.state_hash(system) != before


def test_probe_state_snapshot_allows_only_explicit_diagnostics(
    registry, workspace_tmp
):
    support = load(
        "generic_m20_support_state_allowlist",
        "scripts/generic_m20_probe_support.py",
    )
    runner = load(
        "generic_m20_runner_state_allowlist",
        "scripts/run_v16_generic_m20_fixed_parent_probe.py",
    )
    from multi_dataset_diverse_rl.llm_client import LLMCallResult
    from multi_dataset_diverse_rl.system import CandidateFunnel

    teacher = {
        "failure_pattern": "the solver commits before checking constraints",
        "repair_rule": "Check each explicit constraint before committing.",
        "preservation_rule": "Keep conclusions that pass every explicit check.",
    }
    approved = {"failed_checks": [], "risk_case_ids": [], "feedback": ""}

    async def fake_chat(
        _model, system_prompt, _user_prompt, _temperature, _max_tokens,
        _role, _logical_role=None,
    ):
        if "Check only explicit hard blockers" in system_prompt:
            payload = approved
        elif system_prompt.startswith("Return strict JSON only."):
            payload = {"candidate_prompts": ["candidate probe procedure"]}
        else:
            payload = teacher
        text = json.dumps(payload)
        return LLMCallResult(text, 1, 1, 2, 0.0, "stop")

    case = registry["cases"][0]
    frozen = set(case["assigned_question_hashes"])
    system = support.generation_system(
        case,
        support.G0,
        out_dir=workspace_tmp / "state_allowlist",
        cache_path="",
    )
    system._chat = fake_chat
    core_before = support.state_hash(system)
    diagnostic_before = support.diagnostic_payload(system)
    candidates = asyncio.run(system.propose_candidates(
        int(case["target_agent_id"]),
        support.generation_hashes(support.G0, frozen),
        CandidateFunnel(),
        int(case["source_update_index"]),
    ))
    assert candidates
    assert support.state_hash(system) == core_before
    assert support.diagnostic_payload(system) != diagnostic_before
    context = runner._context_diagnostics(system, support.G0, frozen)
    assert context["generator_assigned_responsibility_hash_count"] == 0
    assert context["responsibility_question_hashes_exposed_to_generator"] == 0
    assert context["member_specific_responsibility_summary_exposed"] is False


def test_source_freeze_rechecks_registry_and_all_definitions(
    registry, workspace_tmp, monkeypatch
):
    module = load(
        "generic_m20_runner_registry_definition_freeze",
        "scripts/run_v16_generic_m20_fixed_parent_probe.py",
    )
    monkeypatch.setattr(module, "ROOT", workspace_tmp)
    source = workspace_tmp / "source.py"
    source.write_text("x = 1\n", encoding="utf-8")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    combined = hashlib.sha256()
    combined.update(b"source.py\0" + source_hash.encode() + b"\n")
    definitions = workspace_tmp / "prep"
    definitions.mkdir()
    source_prep = ROOT / "runs/v16_responsibility_coherence_generic_m20_prep"
    for name in module.FROZEN_DEFINITION_SHA256:
        shutil.copy2(source_prep / name, definitions / name)
    registry_path = definitions / "registry.json"
    disk_registry = json.loads(json.dumps(registry))
    registry_path.write_text(
        json.dumps(disk_registry, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    freeze = {
        "execution_commit": disk_registry["execution_commit"],
        "registry_content_hash": disk_registry["registry_content_hash"],
        "source_freeze_status": "PASS",
        "repo_dirty": False,
        "registry_file_sha256": hashlib.sha256(
            registry_path.read_bytes()
        ).hexdigest(),
        "frozen_definition_sha256": module.FROZEN_DEFINITION_SHA256,
        "working_tree_source_hash": combined.hexdigest(),
        "source_file_count": 1,
        "files": [{"path": "source.py", "sha256": source_hash}],
    }
    monkeypatch.setattr(
        module, "git_head", lambda: disk_registry["execution_commit"]
    )
    monkeypatch.setattr(module, "tracked_source_dirty", lambda: [])
    assert module.source_freeze_gate(
        disk_registry, freeze,
        registry_path=registry_path,
        definition_root=definitions,
        check_inventory=False,
    ) == []
    registry_path.write_text("{}", encoding="utf-8")
    assert "registry_file_hash_mismatch" in module.source_freeze_gate(
        disk_registry, freeze,
        registry_path=registry_path,
        definition_root=definitions,
        check_inventory=False,
    )
    registry_path.write_text(
        json.dumps(disk_registry, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (definitions / "DESIGN_SPEC.md").write_text("changed", encoding="utf-8")
    assert "frozen_definition_hash:DESIGN_SPEC.md" in module.source_freeze_gate(
        disk_registry, freeze,
        registry_path=registry_path,
        definition_root=definitions,
        check_inventory=False,
    )


def test_evaluator_rollout_changes_only_explicit_diagnostics(
    registry, workspace_tmp
):
    support = load(
        "generic_m20_support_evaluator_allowlist",
        "scripts/generic_m20_probe_support.py",
    )
    from multi_dataset_diverse_rl.evaluation.prompt_question import PromptAnswer
    from multi_dataset_diverse_rl.system import CandidateFunnel, CandidateRuntime
    from multi_dataset_diverse_rl.tcs import StudentPromptCandidate

    case = registry["cases"][0]
    evaluator = support.evaluation_system(
        case,
        out_dir=workspace_tmp / "evaluator_allowlist",
        cache_path="",
    )
    target = int(case["target_agent_id"])
    candidate_prompt = "fixed-parent diagnostic candidate"
    prompt_hash = evaluator.prompt_hash(candidate_prompt)

    async def fake_solve(question, _agent_id, _prompt):
        return PromptAnswer(
            answer="A",
            trace="FINAL_ANSWER: A",
            valid=True,
            response_hash=hashlib.sha256(question.encode()).hexdigest(),
            request_identity="offline-test",
        )

    evaluator.solve = fake_solve
    candidate = CandidateRuntime(
        student_candidate=StudentPromptCandidate(candidate_prompt),
        prompt=candidate_prompt,
        prompt_hash=prompt_hash,
        generation=1,
        parent_prompt_hash=evaluator.prompt_hash(
            evaluator.agents[target].current_prompt
        ),
    )
    core_before = support.state_hash(evaluator)
    diagnostic_before = support.diagnostic_payload(evaluator)
    asyncio.run(evaluator.evaluate_candidates(
        target,
        [candidate],
        set(case["assigned_question_hashes"]),
        CandidateFunnel(),
        int(case["source_update_index"]),
    ))
    assert support.state_hash(evaluator) == core_before
    after = support.diagnostic_payload(evaluator)
    assert after != diagnostic_before
    assert after["solver_recovery_observations"] > (
        diagnostic_before["solver_recovery_observations"]
    )


def test_runner_requires_authorization_before_provider_or_output(
    registry, workspace_tmp, monkeypatch
):
    module = load(
        "generic_m20_runner_auth",
        "scripts/run_v16_generic_m20_fixed_parent_probe.py",
    )
    registry_path = workspace_tmp / "registry.json"
    freeze_path = workspace_tmp / "freeze.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    freeze_path.write_text("{}", encoding="utf-8")
    monkeypatch.delenv(
        "GENERIC_M20_FIXED_PARENT_PROBE_AUTHORIZED", raising=False
    )
    args = type("Args", (), {
        "registry": registry_path,
        "source_freeze": freeze_path,
        "out_root": workspace_tmp / "out",
    })()
    with pytest.raises(SystemExit, match="API execution blocked"):
        asyncio.run(module.main_async(args))
    assert not args.out_root.exists()


def test_source_freeze_rehashes_every_file(
    registry, workspace_tmp, monkeypatch
):
    module = load(
        "generic_m20_runner_freeze",
        "scripts/run_v16_generic_m20_fixed_parent_probe.py",
    )
    monkeypatch.setattr(module, "ROOT", workspace_tmp)
    source = workspace_tmp / "source.py"
    source.write_text("x = 1\n", encoding="utf-8")
    relative = "source.py"
    file_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    combined = hashlib.sha256()
    combined.update(relative.encode() + b"\0" + file_hash.encode() + b"\n")
    freeze = {
        "execution_commit": registry["execution_commit"],
        "registry_content_hash": registry["registry_content_hash"],
        "source_freeze_status": "PASS",
        "repo_dirty": False,
        "frozen_definition_sha256": module.FROZEN_DEFINITION_SHA256,
        "working_tree_source_hash": combined.hexdigest(),
        "source_file_count": 1,
        "files": [{"path": relative, "sha256": file_hash}],
    }
    monkeypatch.setattr(module, "git_head", lambda: registry["execution_commit"])
    monkeypatch.setattr(module, "tracked_source_dirty", lambda: [])
    assert module.source_freeze_gate(
        registry, freeze, check_inventory=False
    ) == []
    source.write_text("x = 2\n", encoding="utf-8")
    assert any(
        row.startswith("source_file_hash:")
        for row in module.source_freeze_gate(
            registry, freeze, check_inventory=False
        )
    )


def test_first_success_call_latches_hard_freeze_atomically(
    registry, workspace_tmp, monkeypatch
):
    module = load(
        "generic_m20_runner_latch",
        "scripts/run_v16_generic_m20_fixed_parent_probe.py",
    )
    monkeypatch.setattr(module, "source_freeze_gate", lambda *_, **__: [])
    marker = workspace_tmp / "first_success.json"
    freeze = {"working_tree_source_hash": "a" * 64}
    freeze.update({
        "registry_file_sha256": "b" * 64,
        "frozen_definition_sha256": module.FROZEN_DEFINITION_SHA256,
    })
    latch = module.FirstSuccessfulCallFreeze(registry, freeze, marker)

    class LLM:
        async def chat_result(self, *args, **kwargs):
            return "ok"

    system = type("System", (), {"llm": LLM()})()
    latch.attach(system)
    assert asyncio.run(system.llm.chat_result()) == "ok"
    assert latch.started is True
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["status"] == "HARD"
    assert payload["execution_commit"] == registry["execution_commit"]
    marker_hash = hashlib.sha256(marker.read_bytes()).hexdigest()
    assert asyncio.run(system.llm.chat_result()) == "ok"
    assert hashlib.sha256(marker.read_bytes()).hexdigest() == marker_hash


def test_auditor_rejects_leakage_and_parent_mutation(registry):
    module = load(
        "generic_m20_auditor",
        "scripts/audit_v16_generic_m20_fixed_parent_probe.py",
    )
    cells = []
    for case in registry["cases"]:
        for variant in ("g0_fixed_target_generic", "m20_current_v15"):
            cells.append({
                "case_id": case["case_id"],
                "seed": case["source_seed"],
                "update_index": case["source_update_index"],
                "variant": variant,
                "target_agent_id": case["target_agent_id"],
                "parent_team_hash": case["parent_team_hash"],
                "responsibility_evidence_hash": case[
                    "frozen_responsibility_evidence_hash"
                ],
                "execution_commit": registry["execution_commit"],
                "requested_candidate_count": 2,
                "generation_context": {
                    "context_type": (
                        "AccuracyDiagnosisContext" if variant.startswith("g0")
                        else "SingleLaneDiagnosisContext"
                    ),
                    "responsibility_question_hashes_exposed_to_generator": (
                        1 if variant.startswith("g0") else 2
                    ),
                    "member_specific_responsibility_summary_exposed": (
                        not variant.startswith("g0")
                    ),
                    "coverage_conversion_responsibility_labels_exposed": False,
                    "repair_distance_responsibility_metadata_exposed": False,
                    "frozen_responsibility_hash_overlap_count": (
                        1 if variant.startswith("g0") else 2
                    ),
                    "generator_assigned_responsibility_hash_count": (
                        1 if variant.startswith("g0") else 2
                    ),
                    "forbidden_field_violations": [],
                },
                "evaluation_policy": {
                    "experiment_setting": "experimental_v16_m20_current_v15",
                    "sample_pool_policy": "member_aware_residuals",
                    "stage_a_policy": "matched_all_generated",
                    "candidate_acceptance_policy": (
                        "fixed_peer_monotone_target_or_vote"
                    ),
                    "candidate_ranking_policy": "common_monotone_safe",
                },
                "parent_state_hash_before": "a",
                "parent_state_hash_after": (
                    "b" if variant.startswith("g0") else "a"
                ),
                "generation_parent_state_hash_before": "a",
                "generation_parent_state_hash_after": "a",
                "parent_state_mutation_count": 0,
                "team_prompt_commit_count": 0,
                "optimizer_state_update_count": 0,
                "commit_performed": False,
                "validation_calls": 0,
                "test_calls": 0,
                "funnel": {},
            })
    report = module.audit(registry, {
        "registry_hash": registry["registry_content_hash"],
        "execution_commit": registry["execution_commit"],
        "requested_candidate_count": 32,
        "commit_count": 0,
        "parent_state_mutation_count": 0,
        "optimizer_state_update_count": 0,
        "tracked_source_freeze_hard": True,
        "first_success_source_freeze": {
            "status": "HARD",
            "execution_commit": registry["execution_commit"],
            "registry_content_hash": registry["registry_content_hash"],
            "working_tree_source_hash": "a" * 64,
            "registry_file_sha256": "b" * 64,
            "frozen_definition_sha256": {
                "DESIGN_SPEC.md": "c" * 64,
            },
        },
        "cells": cells,
    }, {
        "working_tree_source_hash": "a" * 64,
        "registry_file_sha256": "b" * 64,
        "frozen_definition_sha256": {"DESIGN_SPEC.md": "c" * 64},
    })
    assert report["gate"] == "FAIL"
    assert report["g0_responsibility_leakage"] == 8
    assert report["parent_mutations"] == 8


def test_freeze_builder_schema_matches_runner_and_hashes_definitions(
    registry, workspace_tmp, monkeypatch
):
    module = load(
        "generic_m20_freeze",
        "scripts/freeze_v16_generic_m20_probe.py",
    )
    prep = workspace_tmp / "prep"
    prep.mkdir()
    source_prep = ROOT / "runs/v16_responsibility_coherence_generic_m20_prep"
    for name in module.DEFINITION_FILES:
        shutil.copy2(source_prep / name, prep / name)
    registry_path = prep / "registry.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    preflight_path = prep / "preflight.json"
    preflight_path.write_text(json.dumps({
        "status": "PASS", "api_calls": 0, "model_calls": 0,
        "g0_responsibility_leakage_count": 0,
    }), encoding="utf-8")
    offline_path = prep / "offline.json"
    offline_path.write_text(json.dumps({
        "status": "PASS", "test_count": 8, "api_calls": 0,
        "model_calls": 0, "validation_calls": 0, "test_calls": 0,
    }), encoding="utf-8")
    source_file = workspace_tmp / "source.py"
    source_file.write_text("x=1\n", encoding="utf-8")
    source_hash = hashlib.sha256(source_file.read_bytes()).hexdigest()
    monkeypatch.setattr(module, "git_head", lambda: registry["execution_commit"])
    monkeypatch.setattr(module, "tracked_source_dirty", lambda: [])
    monkeypatch.setattr(module, "source_manifest", lambda: {
        "source_file_count": 1,
        "working_tree_source_hash": "combined",
        "files": [{"path": "source.py", "sha256": source_hash}],
    })
    freeze, verification, semantics = module.build_freeze(
        registry_path=registry_path,
        prep_root=prep,
        preflight_path=preflight_path,
        offline_verification_path=offline_path,
    )
    assert freeze["source_freeze_status"] == "PASS"
    assert freeze["registry_file_sha256"] == hashlib.sha256(
        registry_path.read_bytes()
    ).hexdigest()
    assert freeze["registry_content_hash_recomputed"] == registry[
        "registry_content_hash"
    ]
    assert set(freeze["frozen_definition_sha256"]) == set(
        module.DEFINITION_FILES
    )
    assert verification["status"] == "PASS"
    assert verification["api_calls"] == 0
    assert semantics["module1_semantics_changed"] is False


def test_analyzer_requires_explicit_source_freeze():
    module = load(
        "generic_m20_analyzer_cli",
        "scripts/analyze_v16_generic_m20_fixed_parent_probe.py",
    )
    source = Path(inspect.getsourcefile(module.main)).read_text(encoding="utf-8")
    assert 'parser.add_argument("--source_freeze", type=Path, required=True)' in source
    assert "v16_responsibility_coherence_generic_m20_prep" not in source
