from __future__ import annotations
import importlib.util,json
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]
def load(name,file):
 spec=importlib.util.spec_from_file_location(name,ROOT/file);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
support=load("m2f_support","scripts/m2f_probe_support.py")
def test_frozen_source_identity(): assert len(support.SOURCE_HASHES)==len(set(support.SOURCE_HASHES))==7
def test_repair_runtime_settings_are_frozen():
    assert support.REPAIR_MAX_TOKENS == 3000
    assert support.MODEL == "qwen3-14b"
    assert support.THINKING is False
def test_parse_single_repair_and_rejects_memorization():
 case={"source_candidate_prompt":"source","repair_evidence":[{"question":"This is a sufficiently long supplied example question body."}],"loss_evidence":[]}
 assert support.parse_repair_output('{"repaired_prompt":"revised general procedure"}',case)=="revised general procedure"
 with pytest.raises(ValueError):support.parse_repair_output('{"repaired_prompt":"This is a sufficiently long supplied example question body."}',case)
def test_loss_decomposition(): assert support.loss_decomposition({"a","b"},{"b","c"})=={"recovered":1,"persistent":1,"new":1}
@pytest.mark.parametrize("ret,sl,rl,w,l,sp,rp,resc,inf,label",[(.79,10,1,7,0,2,0,7,7,"TARGETING_LOST"),(.8,10,10,7,0,2,0,7,7,"COLLATERAL_NOT_REDUCED"),(.8,10,2,7,0,2,3,7,7,"PIVOTAL_COMPATIBILITY_FAILED"),(.8,10,2,7,0,2,1,1,7,"NO_FEASIBILITY_RESCUE"),(.8,10,2,7,0,2,1,2,7,"REPAIR_WORKS")])
def test_classifier_boundaries(ret,sl,rl,w,l,sp,rp,resc,inf,label):assert support.classify(retention=ret,source_loss=sl,repair_loss=rl,collateral_wins=w,collateral_losses=l,source_pivotal=sp,repair_pivotal=rp,rescues=resc,source_infeasible=inf)[0]==label
def test_evaluator_compatibility_is_exact(): assert support.evaluator_compatibility("b74a5d349382532b00df30da77d479f679483297")["status"]=="PASS"

def test_stable_selection_is_margin_descending_then_hash():
    rows = [
        {"question_hash": "b", "parent_plurality_margin": 3},
        {"question_hash": "a", "parent_plurality_margin": 3},
        {"question_hash": "c", "parent_plurality_margin": 4},
    ]
    assert [row["question_hash"] for row in sorted(
        rows, key=lambda row: (-row["parent_plurality_margin"], row["question_hash"])
    )[:2]] == ["c", "a"]

def test_repair_request_has_no_m2e_or_critic_stage():
    case = {
        "parent_prompt": "parent", "source_candidate_prompt": "source",
        "repair_evidence": [], "loss_evidence": [], "numeric_summary": {},
    }
    request = support.build_repair_request(case).lower()
    assert "scoped patch" not in request
    assert "critic" not in request
    assert "source candidate" in request

def test_invalid_source_provenance_and_body_are_holds(tmp_path):
    builder = load("m2f_builder_test", "scripts/build_v16_m2f_probe_registry.py")
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"execution_commit": "wrong"}), encoding="utf-8")
    with pytest.raises(ValueError, match="provenance"):
        builder.build(summary, tmp_path / "missing_registry.json",
                      tmp_path / "missing.sqlite", tmp_path / "missing.csv", "a" * 40)

def test_sanitized_contract_forbids_candidate_bodies():
    packager = load("m2f_packager_scan", "scripts/package_v16_responsibility_reports.py")
    with pytest.raises(ValueError, match="sensitive structured field"):
        packager._scan_structured({"source_candidate_prompt": "private"})

def test_terminal_invalid_delta_uses_nested_competence():
    runner = load("m2f_runner_terminal", "scripts/run_v16_m2f_probe.py")
    from multi_dataset_diverse_rl.candidate_selection import (
        CandidateEvaluation, PromptCompetenceMetrics, TeamOutcomeMetrics,
    )
    from multi_dataset_diverse_rl.member_objectives import MemberGainMetrics
    from multi_dataset_diverse_rl.responsibility import CandidateMarginalContribution, ProtectionContribution
    def evaluation(count):
        return CandidateEvaluation(
            prompt="p", prompt_hash="h",
            competence=PromptCompetenceMetrics(0, 0.0, 0, 0.0, count),
            team_outcome=TeamOutcomeMetrics((), 0, 0.0, (), (), (), 0.0),
            marginal=CandidateMarginalContribution(0, 0, 0, 0.0, 0, 0, 0, 0, 0, 0.0),
            protection=ProtectionContribution(0, 0),
            member_gain=MemberGainMetrics((), (), (), (), 0, 0, 0.0, 0, 0, False, False, 0, 0),
        )
    assert runner.terminal_invalid_delta(evaluation(3), evaluation(1)) == 2
