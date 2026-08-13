from __future__ import annotations

import asyncio
import json
from pathlib import Path

from multi_dataset_diverse_rl.compatibility_repair import (
    build_repair_request,
    parse_repair_output,
    repair_eligible,
)
from multi_dataset_diverse_rl.config import Config
from multi_dataset_diverse_rl.system import PromptEnsembleOptimizationSystem
from multi_dataset_diverse_rl.system import CandidateFunnel, CandidateRuntime
from multi_dataset_diverse_rl.tcs import StudentPromptCandidate
from multi_dataset_diverse_rl.evaluation.fixed_probe import PromptAnswer
from multi_dataset_diverse_rl.persistence.checkpoint import build_checkpoint, restore_checkpoint
from multi_dataset_diverse_rl.persistence.identity import RunIdentity
from multi_dataset_diverse_rl.versions import METHOD_VERSION
from scripts.audit_v16_m2f_online_mechanism_pilot import audit
from scripts.package_v16_m2f_online_mechanism_report import scan


ROOT = Path(__file__).resolve().parents[1]


def identity(setting: str) -> RunIdentity:
    return RunIdentity(
        method_version=METHOD_VERSION, experiment_setting=setting,
        git_commit="commit", git_dirty=False, config_fingerprint="config",
        manifest_sha256="manifest", train_file_sha256="train",
        val_file_sha256="val", test_file_sha256="test",
        train_question_set_hash="train-q", val_question_set_hash="val-q",
        test_question_set_hash="test-q",
    )


def test_online_setting_is_experimental_and_holds_module1_common_safe(tmp_path):
    cfg = Config.from_flat(
        out_dir=str(tmp_path),
        experiment_setting="experimental_v16_m2f_online_compatibility_repair",
    )
    system = PromptEnsembleOptimizationSystem(cfg)
    assert system.protocol.compatibility_repair_enabled is True
    assert system.protocol.module2_evolution_variant == "m20_current_v15"
    assert system.protocol.target_selection_policy == "repairability_adjusted_responsibility"
    assert system.protocol.candidate_acceptance_policy == "fixed_peer_monotone_target_or_vote"
    assert system.protocol.candidates_per_target_branch == 2
    assert system.protocol.target_branch_count == 2


def test_repair_eligibility_requires_gain_collateral_reason_and_evidence():
    assert repair_eligible(
        responsibility_gain_count=1,
        rejection_reasons=["team_vote_regression"],
        loss_evidence_count=1,
    )
    assert not repair_eligible(
        responsibility_gain_count=0,
        rejection_reasons=["team_vote_regression"],
        loss_evidence_count=1,
    )
    assert not repair_eligible(
        responsibility_gain_count=1,
        rejection_reasons=["no_target_or_vote_progress"],
        loss_evidence_count=1,
    )
    assert not repair_eligible(
        responsibility_gain_count=1,
        rejection_reasons=["target_regression"],
        loss_evidence_count=0,
    )


def test_repair_request_contains_actual_paired_evidence():
    request = build_repair_request(
        parent_prompt="parent",
        source_candidate_prompt="source",
        repair_evidence=[{"question_hash": "r", "question": "repair"}],
        loss_evidence=[{"question_hash": "l", "question": "loss"}],
        numeric_summary={"responsibility_gain_count": 1},
    )
    payload = json.loads(request.split("RepairInput:\n", 1)[1])
    assert payload["parent_member_prompt"] == "parent"
    assert payload["source_m20_candidate_prompt"] == "source"
    assert payload["successful_assigned_responsibility_repairs"][0]["question_hash"] == "r"
    assert payload["candidate_specific_competence_losses"][0]["question_hash"] == "l"


def test_repair_parser_rejects_unchanged_and_memorization():
    evidence = [{"question": "This is a sufficiently long private supplied question body."}]
    try:
        parse_repair_output(
            '{"repaired_prompt":"source"}',
            source_candidate_prompt="source",
            supplied_evidence=evidence,
        )
    except ValueError as exc:
        assert "unchanged" in str(exc)
    else:
        raise AssertionError("unchanged online repair accepted")
    try:
        parse_repair_output(
            '{"repaired_prompt":"This is a sufficiently long private supplied question body."}',
            source_candidate_prompt="source",
            supplied_evidence=evidence,
        )
    except ValueError as exc:
        assert "memorizes" in str(exc)
    else:
        raise AssertionError("memorizing online repair accepted")


def test_seed52_preregistration_is_train_only():
    spec = json.loads((
        ROOT / "experiments/v16_m2f_online_mechanism_pilot_seed52_20260813/pilot_preregistration.json"
    ).read_text())
    assert spec["seed"] == 52
    assert spec["updates"] == 8
    assert spec["validation_enabled"] is False
    assert spec["final_test_enabled"] is False
    assert set(spec["analysis_only_metrics"]) == {"critical_net", "oracle_delta", "vote_net"}


def test_auditor_counts_only_infeasible_source_repair_commit(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps({"git_head": "abc"}))
    (run / "run_meta.json").write_text(json.dumps({
        "run_identity": {"git_commit": "abc", "experiment_setting": "experimental_v16_m2f_online_compatibility_repair"},
        "planned_update_count": 8, "completed_update_count": 8,
        "final_test_enabled": False,
    }))
    (run / "final_summary.json").write_text(json.dumps({"selection_summary": {
        "validation_used": False, "validation_evaluation_count": 0,
        "test_evaluation_count": 0,
    }}))
    decisions = []
    commits = []
    branches = []
    for update in range(8):
        candidates = []
        committed = ""
        if update == 0:
            candidates = [
                {"prompt_hash": "source", "candidate_stage": "m20_source", "constraint": {"passed": False}, "repair_plan_hash": ""},
                {"prompt_hash": "repair", "candidate_stage": "compatibility_repair", "constraint": {"passed": True}, "repair_plan_hash": "source"},
            ]
            committed = "repair"
        decisions.append({"update_index": update, "candidates": candidates, "branches": (
            [{"target_agent_id": 0}] if update == 0 else []
        )})
        commits.append({"update_index": update, "parent_team_hash": "parent", "committed_prompt_hash": committed})
    event = {
        "update_index": 0, "source_candidate_hash": "source",
        "source_common_safe": False, "repair_eligible": True,
        "repair_attempted": True, "repair_output_valid": True,
        "repair_feasible": True, "repair_committed": True,
        "repaired_candidate_hash": "repair",
        "target_agent_id": 0, "parent_team_hash": "parent",
    }
    for name, rows in (
        ("candidate_decisions.jsonl", decisions),
        ("dual_target_commit_decisions.jsonl", commits),
        ("dual_target_branch_decisions.jsonl", branches),
        ("online_compatibility_repair_events.jsonl", [event]),
    ):
        (run / name).write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = audit(run, freeze)
    assert result["gate"] == "PASS"
    assert result["execution_commit"] == "abc"
    assert result["repair_attributable_accepted_updates"] == 1


def test_online_repair_rescues_infeasible_source_without_parent_mutation(tmp_path):
    async def solver(question, agent_id, prompt):
        if agent_id != 0:
            answer = "A"
        elif prompt == "source-rule":
            answer = "A" if question == "q0" else "B"
        elif prompt == "repair-rule":
            answer = "A"
        else:
            answer = "B" if question == "q0" else "A"
        return PromptAnswer(answer, f"FINAL_ANSWER: {answer}", True)

    async def optimizer(_system, _user, _temperature, _max_tokens):
        return json.dumps({"repaired_prompt": "repair-rule"})

    async def run():
        system = PromptEnsembleOptimizationSystem(
            Config.from_flat(
                out_dir=str(tmp_path),
                experiment_setting="experimental_v16_m2f_online_compatibility_repair",
            ),
            solver=solver,
            optimizer_chat=optimizer,
        )
        await system.initialize_fixed_probe([
            {"question": question, "answer": "A"}
            for question in ("q0", "q1", "q2")
        ])
        before = system.team_prompt_state_hash()
        candidate = CandidateRuntime(
            StudentPromptCandidate("source-rule"), "source-rule",
            system.prompt_hash("source-rule"), 1,
            system.prompt_hash(system.agents[0].current_prompt),
        )
        funnel = CandidateFunnel()
        _, incumbent, evaluated = await system.evaluate_candidates(
            0, [candidate], {system.fixed_probe.examples[0].question_hash}, funnel
        )
        assert evaluated[0].constraint.passed is False
        repaired = await system._compatibility_repair_candidates(
            target=0,
            assigned_hashes={system.fixed_probe.examples[0].question_hash},
            source_candidates=evaluated,
            incumbent=incumbent,
            update_index=0,
        )
        return system, before, repaired

    system, before, repaired = asyncio.run(run())
    assert len(repaired) == 1
    assert repaired[0].constraint.passed is True
    assert system.team_prompt_state_hash() == before
    event = system.compatibility_repair_events[0]
    assert event["repair_eligible"] is True
    assert event["repair_attempted"] is True
    assert event["repair_feasible"] is True
    assert event["retained_source_responsibility_repairs"] == 1


def test_checkpoint_persists_online_repair_identity_and_events(tmp_path):
    source = PromptEnsembleOptimizationSystem(Config.from_flat(
        out_dir=str(tmp_path / "source"),
        experiment_setting="experimental_v16_m2f_online_compatibility_repair",
    ))
    source.compatibility_repair_events.append({"update_index": 0, "repair_eligible": True})
    source.set_run_identity(identity("experimental_v16_m2f_online_compatibility_repair"))
    asyncio.run(source.initialize_fixed_probe([{"question": "q", "answer": "A"}]))
    source.ensure_responsibility_current()
    payload = build_checkpoint(source, epoch_index=0, update_index=0, training_state={})
    assert payload["compatibility_repair_enabled"] is True
    target = PromptEnsembleOptimizationSystem(Config.from_flat(
        out_dir=str(tmp_path / "source"),
        experiment_setting="experimental_v16_m2f_online_compatibility_repair",
    ))
    target.set_run_identity(identity("experimental_v16_m2f_online_compatibility_repair"))
    asyncio.run(target.initialize_fixed_probe([{"question": "q", "answer": "A"}]))
    restore_checkpoint(target, payload)
    assert target.compatibility_repair_events == source.compatibility_repair_events


def test_checkpoint_rejects_online_repair_setting_mismatch(tmp_path):
    source = PromptEnsembleOptimizationSystem(Config.from_flat(
        out_dir=str(tmp_path / "source"),
        experiment_setting="experimental_v16_m2f_online_compatibility_repair",
    ))
    source.set_run_identity(identity("experimental_v16_m2f_online_compatibility_repair"))
    asyncio.run(source.initialize_fixed_probe([{"question": "q", "answer": "A"}]))
    payload = build_checkpoint(source, epoch_index=0, update_index=0, training_state={})
    target = PromptEnsembleOptimizationSystem(Config.from_flat(
        out_dir=str(tmp_path / "target"),
        experiment_setting="experimental_v16_m20_current_v15",
    ))
    try:
        restore_checkpoint(target, payload)
    except ValueError as exc:
        assert "setting" in str(exc).lower()
    else:
        raise AssertionError("incompatible online repair checkpoint accepted")


def test_report_scanner_rejects_sensitive_fields_and_paths():
    for value in (
        {"parent_prompt": "private"},
        {"safe": "D:/private/run"},
        {"raw_response": "private"},
    ):
        try:
            scan(value)
        except ValueError:
            pass
        else:
            raise AssertionError("sensitive report content accepted")
    scan({"source_candidate_hash": "a" * 64, "repair_committed": True})
