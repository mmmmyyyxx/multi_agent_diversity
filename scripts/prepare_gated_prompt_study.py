"""Offline fresh Validation100 gated study freeze. No API authorization."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from multi_dataset_diverse_rl.research.gated_prompt_governance import prepare
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--destination',type=Path,required=True)
    parser.add_argument('--continuation-root',type=Path)
    parser.add_argument('--parent-inventory-pin',type=Path)
    args=parser.parse_args();print(json.dumps(prepare(ROOT,args.destination,continuation_root=args.continuation_root,parent_inventory_pin=args.parent_inventory_pin),indent=2))
