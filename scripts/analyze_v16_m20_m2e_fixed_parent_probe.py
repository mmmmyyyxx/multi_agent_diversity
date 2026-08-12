from __future__ import annotations
import argparse,csv,json
from pathlib import Path
M20="m20_current_v15";M2E="m2e_scoped_behavioral_patch"
def write_csv(path,rows):
 fields=sorted({k for r in rows for k in r})
 with path.open("w",encoding="utf-8",newline="") as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def metrics(cell):
 cs=cell.get("candidates",[])
 return {"case_id":cell["case_id"],"seed":int(cell["seed"]),"variant":cell["variant"],"requested_candidates":int(cell["requested_candidate_count"]),"evaluated_candidates":len(cs),"feasible_candidates":sum(bool(c.get("constraint",{}).get("passed")) for c in cs),"responsibility_repair":sum(int(c.get("responsibility_residual_gain_count",0)) for c in cs),"nonresponsibility_loss":sum(int(c.get("nonresponsibility_loss_count",0)) for c in cs),"stable_loss":sum(int(c.get("stable_loss_count",0)) for c in cs),"pivotal_loss":sum(int(c.get("pivotal_loss_count",0)) for c in cs),"target_regressions":sum(int(c.get("target_gain",0))<0 for c in cs),"target_gain":sum(int(c.get("target_gain",0)) for c in cs),"F_count":sum(c.get("candidate_geometry")=="F" for c in cs),"student_reached":int(int(cell.get("funnel",{}).get("student_calls",0))>0),"critic_exhausted":int(cell.get("funnel",{}).get("terminal_failure_role")=="critic"),"parent_prefix_identity":sum(bool(c.get("scoped_patch_mechanism",{}).get("parent_prefix_byte_identical")) for c in cs)}
def aggregate(rows,v):
 s=[r for r in rows if r["variant"]==v];keys=("requested_candidates","evaluated_candidates","feasible_candidates","responsibility_repair","nonresponsibility_loss","stable_loss","pivotal_loss","target_regressions","target_gain","F_count","student_reached","critic_exhausted","parent_prefix_identity")
 return {"variant":v,"branches":len(s),**{k:sum(r[k] for r in s) for k in keys}}
def wtl(rows,key,lower=False):
 by={}
 for r in rows:by.setdefault(r["case_id"],{})[r["variant"]]=r[key]
 w=t=l=0
 for pair in by.values():
  d=pair[M2E]-pair[M20];d=-d if lower else d;w+=d>0;t+=d==0;l+=d<0
 return {"wins":w,"ties":t,"losses":l}
def main():
 p=argparse.ArgumentParser();p.add_argument("--run_root",type=Path,required=True);p.add_argument("--protocol_gate",type=Path,required=True);p.add_argument("--source_freeze",type=Path,required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args()
 if a.out.exists():raise SystemExit("analysis output must be fresh")
 gate=json.loads(a.protocol_gate.read_text(encoding="utf-8"));freeze=json.loads(a.source_freeze.read_text(encoding="utf-8"));summary=json.loads((a.run_root/"probe_summary.json").read_text(encoding="utf-8"))
 if gate.get("gate")!="PASS" or freeze.get("source_freeze_status")!="PASS":raise SystemExit("protocol gate and source freeze must PASS")
 rows=[metrics(c) for c in summary["cells"]];variants=[aggregate(rows,v) for v in (M20,M2E)];paired={"responsibility_repair":wtl(rows,"responsibility_repair"),"nonresponsibility_loss":wtl(rows,"nonresponsibility_loss",True),"feasible":wtl(rows,"feasible_candidates"),"target_regression":wtl(rows,"target_regressions",True)};m20,m2e=variants
 retained=m2e["responsibility_repair"]>=0.8*max(1,m20["responsibility_repair"]);reduced=m2e["nonresponsibility_loss"]<m20["nonresponsibility_loss"] and paired["nonresponsibility_loss"]["wins"]>paired["nonresponsibility_loss"]["losses"];value="SUPPORTED" if retained and reduced else "TARGETING_LOST" if not retained else "COLLATERAL_NOT_REDUCED"
 result={"analysis_version":"v16_m20_m2e_analysis_v1","scoped_patch_value":value,"targeting_retained":retained,"collateral_reduced":reduced,"paired":paired,"variants":variants,"execution_commit":summary["execution_commit"]};a.out.mkdir(parents=True);write_csv(a.out/"case_metrics.csv",rows);write_csv(a.out/"variant_metrics.csv",variants);write_csv(a.out/"paired_comparisons.csv",[{"metric":k,**v} for k,v in paired.items()]);(a.out/"analysis_summary.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8");print(json.dumps(result,indent=2))
if __name__=="__main__":main()
