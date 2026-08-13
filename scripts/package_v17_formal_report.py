from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parents[1]
if str(ROOT_PATH) not in sys.path: sys.path.insert(0, str(ROOT_PATH))

from scripts.v17_formal_support import (
    ARMS, EXECUTION_ORDER, REPORT_ROOT, SEEDS, classify_three_seed,
    read_json, recursive_sanitize, sha256_file, write_json,
)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows: raise ValueError(f"empty report table: {path.name}")
    with path.open("w",encoding="utf-8",newline="") as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main() -> None:
    p=argparse.ArgumentParser();p.add_argument("--run_root",type=Path,required=True);p.add_argument("--out",type=Path,default=REPORT_ROOT);a=p.parse_args()
    if a.out.exists(): raise FileExistsError("fresh report directory required")
    for name in ("train_protocol_gate.json","validation_gate.json","test_gate.json","pre_test_seal.json"):
        if read_json(a.run_root/name).get("gate")!="PASS": raise RuntimeError(f"required gate is not PASS: {name}")
    train=read_json(a.run_root/"train_protocol_gate.json");validation=read_json(a.run_root/"validation_gate.json");test=read_json(a.run_root/"test_gate.json")
    train_by={(r["seed"],r["arm"]):r for r in train["cells"]};val_by={(r["seed"],next(k for k,v in ARMS.items() if v==r["setting"])):r for r in validation["rows"]};test_by={(r["seed"],next(k for k,v in ARMS.items() if v==r["setting"])):r for r in test["rows"]}
    rows=[]
    for seed in SEEDS:
      for arm in ARMS:
        tr=train_by[(seed,arm)];v=val_by[(seed,arm)];t=test_by[(seed,arm)]
        rows.append({"arm":arm,"seed":seed,"train_vote_accuracy":tr["final_train_vote_accuracy"],"validation_vote_accuracy":v["vote_accuracy"],"test_vote_accuracy":t["vote_accuracy"],"test_correct_count":t["vote_correct_count"],"g_min":tr["g_min"],"g_sum":tr["g_sum"],"accepted_updates":tr["accepted_updates"],"training_provider_calls":tr["provider_calls"],"training_total_tokens":tr["total_tokens"],"validation_provider_calls":v["provider_calls"],"test_provider_calls":t["provider_calls"],"generic_revision_committed":tr["generic_revision_committed"],"m2f_repair_committed":tr["repair_committed"]})
    aggregates=[]
    for arm in ARMS:
      selected=[r for r in rows if r["arm"]==arm]
      aggregates.append({"arm":arm,"mean_train_vote_accuracy":statistics.mean(r["train_vote_accuracy"] for r in selected),"mean_validation_vote_accuracy":statistics.mean(r["validation_vote_accuracy"] for r in selected),"mean_test_vote_accuracy":statistics.mean(r["test_vote_accuracy"] for r in selected),"median_test_vote_accuracy":statistics.median(r["test_vote_accuracy"] for r in selected),"mean_g_min":statistics.mean(r["g_min"] for r in selected),"mean_g_sum":statistics.mean(r["g_sum"] for r in selected),"total_accepted_updates":sum(r["accepted_updates"] for r in selected),"total_training_provider_calls":sum(r["training_provider_calls"] for r in selected),"total_training_tokens":sum(r["training_total_tokens"] for r in selected)})
    pairs={"C01":("S0","S1"),"C12":("S1","S2"),"C23":("S2","S3"),"C34":("S3","S4"),"C14":("S1","S4"),"C04":("S0","S4")};contrasts=[];classifiers={}
    by={(r["seed"],r["arm"]):r for r in rows}
    for name,(left,right) in pairs.items():
      ds=[]
      for seed in SEEDS:
        delta=by[(seed,right)]["test_vote_accuracy"]-by[(seed,left)]["test_vote_accuracy"];ds.append(delta);contrasts.append({"contrast":name,"seed":seed,"left":left,"right":right,"delta":delta,"result":"WIN" if delta>0 else "LOSS" if delta<0 else "TIE"})
      classifiers[name]=classify_three_seed(ds)
    a.out.mkdir(parents=True)
    write_csv(a.out/"arm_seed_results.csv",rows);write_csv(a.out/"arm_aggregates.csv",aggregates);write_csv(a.out/"paired_test_contrasts.csv",contrasts)
    write_csv(a.out/"validation_results.csv",[{
      "arm":r["arm"],"seed":r["seed"],"vote_accuracy":r["validation_vote_accuracy"]
    } for r in rows])
    write_csv(a.out/"test_results.csv",[{
      "arm":r["arm"],"seed":r["seed"],"vote_accuracy":r["test_vote_accuracy"],
      "correct_count":r["test_correct_count"]
    } for r in rows])
    trajectories=[];mechanisms=[];compute=[]
    for tr in train["cells"]:
      for point in tr["training_trajectory"]: trajectories.append({"arm":tr["arm"],"seed":tr["seed"],**point})
      mechanisms.append({key:tr[key] for key in ("arm","seed","generic_revision_attempted","generic_revision_feasible","generic_revision_committed","repair_eligible","repair_attempted","repair_valid","repair_feasible","repair_committed")})
      compute.append({key:tr[key] for key in ("arm","seed","provider_calls","prompt_tokens","completion_tokens","total_tokens")})
    write_csv(a.out/"training_trajectories.csv",trajectories);write_csv(a.out/"mechanism_metrics.csv",mechanisms);write_csv(a.out/"compute_metrics.csv",compute)
    write_json(a.out/"formal_classifier.json",classifiers)
    write_json(a.out/"protocol_gate.json",{"train":"PASS","validation":"PASS","test":"PASS","paper_untouched_test":False,"historical_test_exposure":True})
    write_json(a.out/"dataset_provenance.json",read_json(a.run_root/"phase_a"/"source_freeze_manifest.json")["dataset_freeze"])
    seal=read_json(a.run_root/"pre_test_seal.json");write_json(a.out/"pre_test_seal_sanitized.json",{"gate":seal["gate"],"execution_commit":seal["execution_commit"],"source_tree_hash":seal["source_tree_hash"],"final_state_count":seal["final_state_count"],"validation_results_already_observed":True,"method_changed_after_validation":False,"test_calls_so_far":0})
    freeze=read_json(a.run_root/"phase_a"/"source_freeze_manifest.json")
    write_json(a.out/"source_freeze_sanitized.json",{
      "source_freeze_status":freeze["source_freeze_status"],
      "git_head":freeze["git_head"],"working_tree_source_hash":freeze["working_tree_source_hash"],
      "source_file_count":freeze["source_file_count"],"definition_sha256":freeze["definition_sha256"],
      "mechanism_hashes":freeze["mechanism_hashes"]
    })
    write_json(a.out/"emergent_complementarity_summary.json",{
      "status":"analysis_not_optimization","derived_without_extra_model_calls":True,
      "oracle_metrics_available_in_private_phase_evaluations":True,
      "critical_competence_metrics_role":"analysis_only"
    })
    (a.out/"README.md").write_text("# V17 Formal Five-Arm Results\n\nFrozen-method comparison on Seeds 56-58. Validation is a post-training audit and did not select states. The test split was historically exposed during development; these results are not an untouched paper-heldout claim.\n",encoding="utf-8")
    problems=[]
    for path in a.out.iterdir():
      if path.suffix==".json": problems.extend(recursive_sanitize(read_json(path)))
      elif path.suffix in {".csv",".md"}:
        text=path.read_text(encoding="utf-8");problems.extend(recursive_sanitize(text))
    if problems: raise RuntimeError("sanitization failed: "+",".join(sorted(set(problems))))
    manifest={path.name:{"sha256":sha256_file(path),"size":path.stat().st_size} for path in sorted(a.out.iterdir())};write_json(a.out/"sanitized_manifest.json",manifest)
    print(json.dumps({"status":"PASS","files":len(manifest)+1,"sanitization":"PASS"},indent=2))


if __name__=="__main__":main()
