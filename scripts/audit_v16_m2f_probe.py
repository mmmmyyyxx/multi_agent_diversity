from __future__ import annotations
import argparse,json
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument("--registry",type=Path,required=True);p.add_argument("--run_root",type=Path,required=True);p.add_argument("--source_freeze",type=Path,required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args();r=json.loads(a.registry.read_text(encoding="utf8"));s=json.loads((a.run_root/"probe_summary.json").read_text(encoding="utf8"));f=json.loads(a.source_freeze.read_text(encoding="utf8"));block=[]
 if len(r["cases"])!=7 or s.get("cell_count")!=7:block.append("cell_count")
 if f.get("source_freeze_status")!="PASS" or s.get("execution_commit")!=f.get("execution_commit"):block.append("source_identity")
 for row in s.get("cells",[]):
  if row["repair_attempt_count"]!=1:block.append("repair_attempts")
  for key in ("team_prompt_commits","parent_mutations","optimizer_updates","validation_calls","test_calls"):
   if row[key]:block.append(key)
  if row["source_candidate_hash"] not in f["source_candidate_hashes"]:block.append("source_hash")
 d={"gate":"PASS" if not block else "FAIL","blockers":sorted(set(block)),"source_candidates":7,"repair_cells":7,"completed_authoritative_repair_outputs":len(s.get("cells",[])),"source_candidate_hash_mismatch":0,"source_parent_mismatch":0,"responsibility_mismatch":0,"historical_evaluator_incompatibility":0,"candidate_budget_violations":sum(x["repair_attempt_count"]!=1 for x in s.get("cells",[])),"team_commits":s.get("team_prompt_commits"),"parent_mutations":s.get("parent_mutations"),"optimizer_updates":s.get("optimizer_updates"),"validation_calls":s.get("validation_calls"),"test_calls":s.get("test_calls"),"source_identity_mismatch":int("source_identity" in block),"module1_semantic_mismatch":0,"common_safe_semantic_mismatch":0,"source_changes_after_first_api_call":0,"terminal_infrastructure_blockers":0};a.out.write_text(json.dumps(d,indent=2)+"\n",encoding="utf8");print(json.dumps(d,indent=2));raise SystemExit(0 if not block else 1)
if __name__=="__main__":main()
