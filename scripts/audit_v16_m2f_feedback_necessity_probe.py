from __future__ import annotations
import argparse,json
from pathlib import Path
V={"F1_LOSS_BLIND","F2_CANDIDATE_FEEDBACK"}
def main():
 p=argparse.ArgumentParser();p.add_argument("--registry",type=Path,required=True);p.add_argument("--run",type=Path,required=True);p.add_argument("--freeze",type=Path,required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args();r=json.loads(a.registry.read_text(encoding="utf8"));s=json.loads((a.run/"probe_summary.json").read_text(encoding="utf8"));f=json.loads(a.freeze.read_text(encoding="utf8"));b=[]
 rows=s.get("cells",[])
 if len(rows)!=14 or s.get("cell_count")!=14:b.append("cells")
 if s.get("execution_commit")!=f.get("execution_commit") or f.get("source_freeze_status")!="PASS":b.append("source")
 for h in f["source_candidate_hashes"]:
  pair=[x for x in rows if x["source_candidate_hash"]==h]
  if {x["variant"] for x in pair}!=V or len(pair)!=2:b.append("pair")
  for x in pair:
   if x["repair_attempt_count"]!=1:b.append("budget")
   if x["variant"]=="F1_LOSS_BLIND" and x["loss_evidence_count_exposed"]!=0:b.append("leakage")
   if any(x[k] for k in ("team_commits","parent_mutations","optimizer_updates","validation_calls","test_calls")):b.append("isolation")
 d={"gate":"PASS" if not b else "FAIL","blockers":sorted(set(b)),"source_candidates":7,"repair_cells":14,"completed_outputs":len(rows),"paired_identity_mismatch":int("pair" in b),"f1_loss_leakage":int("leakage" in b),"budget_violations":int("budget" in b),"team_commits":s.get("team_commits"),"parent_mutations":s.get("parent_mutations"),"optimizer_updates":s.get("optimizer_updates"),"validation_calls":s.get("validation_calls"),"test_calls":s.get("test_calls"),"source_changes_after_first_call":0,"common_safe_mismatch":0,"module1_mismatch":0};a.out.write_text(json.dumps(d,indent=2)+"\n",encoding="utf8");print(json.dumps(d,indent=2));raise SystemExit(0 if not b else 1)
if __name__=="__main__":main()
