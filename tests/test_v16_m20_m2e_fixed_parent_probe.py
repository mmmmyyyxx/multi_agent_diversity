from __future__ import annotations
import importlib.util,json
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(spec);assert spec.loader;spec.loader.exec_module(m);return m
def test_registry_reuses_exact_eight_parents_and_balances_order():
 m=load("m2e_registry","scripts/build_v16_m20_m2e_probe_registry.py");r=m.build_registry("0"*40)
 assert r["variants"]==[m.M20,m.M2E] and len(r["cases"])==8
 assert all(c["cell_order"]==([m.M20,m.M2E] if i%2==0 else [m.M2E,m.M20]) for i,c in enumerate(r["cases"]))
 old=json.loads((ROOT/"runs/v16_responsibility_coherence_generic_m20_retry1_prep/probe_registry_private.json").read_text(encoding="utf-8"))
 keys=("case_id","parent_team_hash","target_agent_id","assigned_question_hashes")
 assert [[c[k] for k in keys] for c in r["cases"]]==[[c[k] for k in keys] for c in old["cases"]]
def test_offline_preflight_locks_context_parent_and_append_mechanism(tmp_path):
 b=load("m2e_registry2","scripts/build_v16_m20_m2e_probe_registry.py");p=load("m2e_preflight","scripts/preflight_v16_m20_m2e_probe.py");r=b.build_registry("0"*40);out=p.preflight(r,scratch=tmp_path)
 assert out["status"]=="PASS" and out["api_calls"]==out["model_calls"]==0
 assert all(x["parent_prefix_byte_identical"] for x in out["mechanisms"])
def test_protocol_auditor_rejects_bad_scoped_mechanism():
 a=load("m2e_auditor","scripts/audit_v16_generic_m20_fixed_parent_probe.py");b=load("m2e_registry3","scripts/build_v16_m20_m2e_probe_registry.py");r=b.build_registry("0"*40)
 cells=[]
 for c in r["cases"]:
  for v in r["variants"]:
   cells.append({"case_id":c["case_id"],"variant":v,"seed":c["source_seed"],"update_index":c["source_update_index"],"parent_team_hash":c["parent_team_hash"],"target_agent_id":c["target_agent_id"],"responsibility_evidence_hash":c["frozen_responsibility_evidence_hash"],"execution_commit":r["execution_commit"],"registry_content_hash":r["registry_content_hash"],"requested_candidate_count":2,"evaluation_policy":{},"parent_state_hash_before":"x","parent_state_hash_after":"x","generation_parent_state_hash_before":"x","generation_parent_state_hash_after":"x","parent_state_mutation_count":0,"team_prompt_commit_count":0,"optimizer_state_update_count":0,"commit_performed":False,"validation_calls":0,"test_calls":0,"funnel":{},"candidates":([{"scoped_patch_mechanism":{"enabled":False}}] if v==b.M2E else [])})
 summary={"cells":cells,"registry_hash":r["registry_content_hash"],"execution_commit":r["execution_commit"],"requested_candidate_count":32,"tracked_source_freeze_hard":True,"first_success_source_freeze":{"status":"HARD","execution_commit":r["execution_commit"],"registry_content_hash":r["registry_content_hash"]},"commit_count":0,"parent_state_mutation_count":0,"optimizer_state_update_count":0}
 report=a.audit(r,summary)
 assert report["gate"]=="FAIL" and report["scoped_patch_mechanism_violations"]==8


def test_candidate_payload_records_append_only_mechanism():
    runner=load("m2e_runner_payload","scripts/run_v16_generic_m20_fixed_parent_probe.py")
    runner.asdict=lambda _value: {}
    trigger="When two interpretations remain plausible";behavior="Compare decisive evidence before committing.";parent="parent bytes"
    prompt=parent+"\n\n[Responsibility-specific conditional refinement]\n"+f"When {trigger}:\n    {behavior}\n\nOutside this condition, follow the original procedure unchanged."
    row=SimpleNamespace(
        final_evaluation=SimpleNamespace(prompt=prompt),
        constraint=SimpleNamespace(passed=True),module2_diagnostics={},prompt_hash="a"*64,
        student_candidate=SimpleNamespace(trigger_condition=trigger,localized_behavior=behavior),
    )
    payload=runner._candidate_payload(row,responsibility_effects={})
    assert payload["scoped_patch_mechanism"]["parent_prefix_byte_identical"] is True
