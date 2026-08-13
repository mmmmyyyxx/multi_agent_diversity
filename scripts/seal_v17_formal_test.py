from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parents[1]
if str(ROOT_PATH) not in sys.path: sys.path.insert(0, str(ROOT_PATH))

from scripts.v17_formal_support import ARMS, EXECUTION_ORDER, EXPERIMENT_ROOT, SEEDS, read_json, sha256_file, tracked_source_inventory, write_json


def main() -> None:
    p=argparse.ArgumentParser();p.add_argument("--run_root",type=Path,required=True);p.add_argument("--freeze",type=Path,required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args()
    freeze=read_json(a.freeze);train=read_json(a.run_root/"train_protocol_gate.json");validation=read_json(a.run_root/"validation_gate.json")
    _,source_hash=tracked_source_inventory();errors=[]
    if train.get("gate")!="PASS": errors.append("train_gate")
    if validation.get("gate")!="PASS": errors.append("validation_gate")
    if source_hash!=freeze.get("working_tree_source_hash"): errors.append("source_hash")
    states=[]
    for seed in SEEDS:
      for arm in EXECUTION_ORDER[seed]:
        setting=ARMS[arm];run=a.run_root/f"seed{seed}"/"disambiguation_qa"/f"{setting}_seed{seed}"
        meta=read_json(run/"run_meta.json")
        states.append({"seed":seed,"arm":arm,"final_state_hash":meta["final_state_selection"]["selected_team_prompt_state_hash"],"checkpoint_sha256":sha256_file(run/"training_checkpoint.json")})
    payload={"seal_version":"v17_pre_test_seal_v1","gate":"PASS" if not errors else "FAIL","errors":errors,"execution_commit":freeze["git_head"],"source_tree_hash":source_hash,"final_states":states,"final_state_count":len(states),"dataset_freeze":freeze["dataset_freeze"],"definition_sha256":freeze["definition_sha256"],"arm_definitions_hash":freeze["arm_definitions_hash"],"classifier_hash":freeze["classifier_hash"],"validation_results_already_observed":True,"method_changed_after_validation":False,"test_calls_so_far":0}
    if a.out.exists(): raise FileExistsError("fresh pre-test seal required")
    write_json(a.out,payload);print(json.dumps({"gate":payload["gate"],"errors":errors,"final_state_count":len(states)},indent=2));raise SystemExit(0 if not errors else 1)


if __name__=="__main__":main()
