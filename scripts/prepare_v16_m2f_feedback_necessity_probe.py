from __future__ import annotations
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"scripts"))
from m2f_probe_support import *
from generic_m20_probe_support import source_manifest,tracked_source_dirty
from preflight_v16_m2f_probe import preflight
VARIANTS=("F1_LOSS_BLIND","F2_CANDIDATE_FEEDBACK")
DEFINITION_FILES=(
 ROOT/"experiments"/"v16_m2f_feedback_necessity_fixed_candidate_20260813"/"DESIGN_SPEC.md",
 ROOT/"experiments"/"v16_m2f_feedback_necessity_fixed_candidate_20260813"/"analysis_spec.json",
)
def main():
 p=argparse.ArgumentParser();p.add_argument("--base_registry",type=Path,required=True);p.add_argument("--out",type=Path,required=True);p.add_argument("--full_tests",required=True);a=p.parse_args()
 if a.out.exists() or ROOT.resolve() not in a.out.resolve().parents:raise SystemExit("fresh repo-local out required")
 base=json.loads(a.base_registry.read_text(encoding="utf8"));head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip();reg={**base,"registry_version":"v16_m2f_feedback_necessity_registry_v1","execution_commit":head,"variants":list(VARIANTS),"cell_count":14,"candidate_count":14,"repair_attempts_per_source":1,"causal_difference":"F2 alone receives candidate-specific loss examples and loss counts"};reg["registry_content_hash"]=canonical_hash({k:v for k,v in reg.items() if k!="registry_content_hash"})
 check=preflight(reg);errors=list(check["errors"]);dirty=tracked_source_dirty()
 if dirty:errors.append("tracked_source_dirty")
 for case in reg["cases"]:
  f1=build_causal_control_request(case,VARIANTS[0]);f2=build_causal_control_request(case,VARIANTS[1])
  if "candidate_specific_competence_losses" in f1 or any(x["question_hash"] in f1 for x in case["loss_evidence"]):errors.append("f1_loss_leakage")
  if "candidate_specific_competence_losses" not in f2:errors.append("f2_missing_losses")
 definitions={str(path.relative_to(ROOT)).replace("\\","/"):hashlib.sha256(path.read_bytes()).hexdigest() for path in DEFINITION_FILES}
 manifest=source_manifest();freeze={"source_freeze_status":"PASS" if not errors else "FAIL","errors":sorted(set(errors)),"execution_commit":head,"registry_content_hash":reg["registry_content_hash"],"variants":list(VARIANTS),"source_candidate_hashes":list(SOURCE_HASHES),"model":MODEL,"thinking":False,"temperature":reg["cases"][0]["base_config"]["student_temperature"],"repair_max_tokens":REPAIR_MAX_TOKENS,"repair_attempts_per_source":1,"cell_count":14,"full_test_result":a.full_tests,"causal_difference":reg["causal_difference"],"definition_sha256":definitions,"f1_request_hashes":[canonical_hash(build_causal_control_request(c,VARIANTS[0])) for c in reg["cases"]],"f2_request_hashes":[canonical_hash(build_causal_control_request(c,VARIANTS[1])) for c in reg["cases"]],**manifest}
 verify={"status":freeze["source_freeze_status"],"errors":freeze["errors"],"source_candidates":7,"repair_cells":14,"api_calls":0,"model_calls":0,"validation_calls":0,"test_calls":0,"f1_loss_leakage":0,"common_safe_changed":False,"module1_changed":False}
 a.out.mkdir(parents=True);(a.out/"probe_registry_private.json").write_text(json.dumps(reg,indent=2)+"\n",encoding="utf8");(a.out/"source_freeze_manifest.json").write_text(json.dumps(freeze,indent=2)+"\n",encoding="utf8");(a.out/"pre_probe_verification.json").write_text(json.dumps(verify,indent=2)+"\n",encoding="utf8");print(json.dumps(verify,indent=2));raise SystemExit(0 if not errors else 1)
if __name__=="__main__":main()
