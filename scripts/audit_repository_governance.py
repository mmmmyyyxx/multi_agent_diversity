"""Offline governance audit; does not execute experiments or providers."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from multi_dataset_diverse_rl.governance.repository import (
    audit_repository,build_report_index,report_index_markdown,render_lineage_v2,
)


def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument('--workspace',type=Path,default=ROOT)
    parser.add_argument('--generate',action='store_true')
    parser.add_argument('--check-generated',action='store_true')
    parser.add_argument('--out',type=Path)
    args=parser.parse_args();root=args.workspace.resolve()
    if args.generate:
        index=build_report_index(root)
        outputs={'docs/experiments/LINEAGE.md':render_lineage_v2(root),
                 'reports/INDEX.md':report_index_markdown(index),
                 'reports/index.json':json.dumps(index,indent=2)+'\n'}
        for path,text in outputs.items():(root/path).write_bytes(text.encode())
    result=audit_repository(root,args.check_generated or args.generate)
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_bytes((json.dumps(result,indent=2)+'\n').encode())
    print(json.dumps({k:v for k,v in result.items() if k!='source_identity'},indent=2))
    return 0 if result['ok'] else 1


if __name__=='__main__':raise SystemExit(main())
