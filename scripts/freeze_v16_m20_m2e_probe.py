from __future__ import annotations
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"scripts"))
from generic_m20_probe_support import source_manifest,tracked_source_dirty
from run_v16_generic_m20_fixed_parent_probe import canonical_registry_hash,validate_registry_contract
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument("--registry",type=Path,required=True);p.add_argument("--definition",type=Path,required=True);p.add_argument("--preflight",type=Path,required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args()
 if a.out.exists():raise SystemExit("freeze output must be fresh")
 r=json.loads(a.registry.read_text(encoding="utf-8"));pre=json.loads(a.preflight.read_text(encoding="utf-8"));head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip();dirty=tracked_source_dirty();errors=validate_registry_contract(r)
 if r.get("execution_commit")!=head:errors.append("execution_commit")
 if dirty:errors.append("tracked_source_dirty")
 if pre.get("status")!="PASS" or any(int(pre.get(k,-1)) for k in ("api_calls","model_calls","validation_calls","test_calls")):errors.append("preflight")
 if r.get("registry_content_hash")!=canonical_registry_hash(r):errors.append("registry_hash")
 manifest={"freeze_version":"v16_m20_m2e_source_freeze_v1","source_freeze_status":"PASS" if not errors else "FAIL","errors":sorted(set(errors)),"execution_commit":head,"repo_dirty":bool(dirty),"registry_file_sha256":sha(a.registry),"registry_content_hash":r["registry_content_hash"],"frozen_definition_sha256":{a.definition.name:sha(a.definition)},"case_count":8,"cell_count":16,"candidate_count":32,**source_manifest()}
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8");print(json.dumps({"status":manifest["source_freeze_status"],"errors":manifest["errors"],"execution_commit":head},indent=2));raise SystemExit(0 if not errors else 1)
if __name__=="__main__":main()
