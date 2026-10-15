"""Read sealed historical candidates under a pre-import offline network guard."""
import argparse,json,socket,sys
from pathlib import Path

def main():
    if socket.socket.connect.__module__!='sitecustomize':
        raise RuntimeError('OFFLINE_NETWORK_GUARD_REQUIRED')
    root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
    parser=argparse.ArgumentParser()
    parser.add_argument('--index',type=Path,default=root/'reports/historical_candidate_guard_audit_20261015/evidence_index.json')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    from multi_dataset_diverse_rl.diagnostics.structural_candidate_replay import compare_candidates
    value=compare_candidates(root,args.index)
    assert value['summary']['draws']==493
    assert value['summary']['confirmed_false_rejections']==value['summary']['confirmed_false_now_eligible']==10
    args.out.parent.mkdir(parents=True,exist_ok=True)
    if args.out.exists():raise ValueError('OUTPUT_ALREADY_EXISTS')
    args.out.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(value['summary'],ensure_ascii=True))
    print(json.dumps(dict(network_attempts=__import__('sitecustomize').network_attempt_count())))

if __name__=='__main__':main()
