from __future__ import annotations
import argparse,csv,json
from pathlib import Path
from m2f_probe_support import classify
def analyze(summary):
 rows=summary["cells"]
 sg=sum(len(r["source_responsibility_gain_hashes"]) for r in rows);ret=sum(r["retained_source_repairs"] for r in rows);rg=sum(r["repair_metrics"]["responsibility_gain_count"] for r in rows)
 sl=sum(sum(len(v) for v in r["source_loss_hashes"].values()) for r in rows);rl=sum(r["repair_metrics"]["nonresponsibility_loss_count"] for r in rows)
 sp=sum(len(r["source_loss_hashes"]["pivotal"])+len(r["source_loss_hashes"]["unique"]) for r in rows);rp=sum(r["repair_metrics"]["pivotal_loss_count"] for r in rows)
 cw=sum(r["repair_metrics"]["nonresponsibility_loss_count"]<sum(len(v) for v in r["source_loss_hashes"].values()) for r in rows);cl=sum(r["repair_metrics"]["nonresponsibility_loss_count"]>sum(len(v) for v in r["source_loss_hashes"].values()) for r in rows)
 infeasible=sum(not r["source_common_safe"] for r in rows);rescues=sum(r["compatibility_rescue"] for r in rows);label,criteria=classify(retention=ret/sg,source_loss=sl,repair_loss=rl,collateral_wins=cw,collateral_losses=cl,source_pivotal=sp,repair_pivotal=rp,rescues=rescues,source_infeasible=infeasible)
 return {"analysis_version":"v16_m2f_analysis_v1","frozen_classifier":label,"promising_for_train_pilot":label=="REPAIR_WORKS","source_total_responsibility_repairs":sg,"retained_source_repairs":ret,"repair_total_responsibility_repairs":rg,"aggregate_targeting_retention":ret/sg,"source_nonresponsibility_loss":sl,"repair_nonresponsibility_loss":rl,"collateral_wtl":{"wins":cw,"ties":7-cw-cl,"losses":cl},"source_pivotal_loss":sp,"repair_pivotal_loss":rp,"compatibility_rescues":rescues,"source_infeasible":infeasible,"criteria":criteria,"recovered_collateral":sum(r["loss_decomposition"]["recovered"] for r in rows),"persistent_collateral":sum(r["loss_decomposition"]["persistent"] for r in rows),"new_collateral":sum(r["loss_decomposition"]["new"] for r in rows)}
def main():
 p=argparse.ArgumentParser();p.add_argument("--summary",type=Path,required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args();d=analyze(json.loads(a.summary.read_text(encoding="utf8")));a.out.write_text(json.dumps(d,indent=2)+"\n",encoding="utf8");print(json.dumps(d,indent=2))
if __name__=="__main__":main()
