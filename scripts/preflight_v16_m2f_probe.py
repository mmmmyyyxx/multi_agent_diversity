from __future__ import annotations
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"scripts"))
from m2f_probe_support import *
def preflight(reg):
 errors=[]
 if len(reg.get("cases",()))!=7 or tuple(c["source_candidate_hash"] for c in reg["cases"])!=SOURCE_HASHES:errors.append("source_identity")
 for c in reg.get("cases",()):
  if sha_text(c["source_candidate_prompt"])!=c["source_candidate_hash"]:errors.append("source_hash")
  if len(c["loss_evidence"])>len(c["all_source_loss_hashes"]["pivotal"])+len(c["all_source_loss_hashes"]["unique"])+2:errors.append("stable_cap")
  if set(c["source_responsibility_gain_hashes"])!={x["question_hash"] for x in c["repair_evidence"]}:errors.append("repair_evidence")
  stable=[x for x in c["loss_evidence"] if x["parent_competence_role"]=="stable"]
  if stable!=sorted(stable,key=lambda x:(-x["parent_plurality_margin"],x["question_hash"]))[:2]:errors.append("stable_order")
  request=build_repair_request(c)
  if "scoped patch" in request.lower() or "critic" in request.lower():errors.append("wrong_mechanism")
 compatibility=evaluator_compatibility(reg["execution_commit"])
 if compatibility["status"]!="PASS":errors.append("evaluator_compatibility")
 return {"status":"PASS" if not errors else "FAIL","errors":sorted(set(errors)),"source_candidates_recovered":7,"source_hash_mismatch":errors.count("source_hash"),"repair_cells":7,"repair_attempts_per_source":1,"model":MODEL,"thinking":False,"module1_changed":False,"common_safe_changed":False,"api_calls":0,"model_calls":0,"validation_calls":0,"test_calls":0,"historical_evaluator_compatibility":compatibility}
def main():
 p=argparse.ArgumentParser();p.add_argument("--registry",type=Path,required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args();d=preflight(json.loads(a.registry.read_text(encoding="utf8")));a.out.write_text(json.dumps(d,indent=2)+"\n",encoding="utf8");print(json.dumps(d,indent=2));raise SystemExit(0 if d["status"]=="PASS" else 1)
if __name__=="__main__":main()
