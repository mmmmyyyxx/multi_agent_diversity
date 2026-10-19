"""Zero-network offline oracle-repair diagnostic; never an experiment runner."""
import argparse
import json
from pathlib import Path
import socket
import sys


def main():
    if socket.socket.connect.__module__ != 'sitecustomize':
        raise RuntimeError('PREIMPORT_OFFLINE_NETWORK_GUARD_REQUIRED')
    root=Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root))
    from multi_dataset_diverse_rl.diagnostics.repair_path import exhaustive_audit,repair_game
    from multi_dataset_diverse_rl.diagnostics.repair_path_replay import check_inputs,historical_replay,sha256
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,default=root/'experiments/manifests/repair_path_responsibility_offline_v1.json')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():raise ValueError('OUTPUT_ALREADY_EXISTS')
    manifest=json.loads(args.manifest.read_text(encoding='utf-8'))
    if manifest['real_api_calls']!=0 or manifest['validation_raw_reads']!=0 or manifest['test_raw_reads']!=0:
        raise ValueError('OFFLINE_SCOPE_REQUIRED')
    manifest_hash=sha256(args.manifest)
    check_inputs(root,manifest)
    exact=exhaustive_audit()
    replay=historical_replay(root/manifest['trajectory'])
    for key,value in manifest['expected_inventory'].items():
        if replay['counters'][key]!=value:raise ValueError('HISTORICAL_INVENTORY_MISMATCH: '+key)
    # Reuse only the previous report's unchanged, sanitized arithmetic.
    prior=json.loads((root/manifest['prior_mechanism_summary']).read_text(encoding='utf-8'))
    prior_full=prior['full_candidates']
    actual=replay['full_candidates']
    expected=[(x['opportunity'],x['full_vote_delta'],x['full_target_delta'],x['vote_fixed'],x['vote_lost']) for x in prior_full]
    observed=[(x['opportunity'],x['vote_delta'],x['target_delta'],x['vote_fixed'],x['vote_lost']) for x in actual]
    if observed!=expected:raise ValueError('PRIOR_AUDITED_FULL_ARITHMETIC_MISMATCH')
    check_inputs(root,manifest)
    if sha256(args.manifest)!=manifest_hash:raise ValueError('MANIFEST_CHANGED_DURING_READ')
    examples={}
    for name,classes in [('all_wrong_same',(1,1,1,1,1)),('two_correct_concentrated',(0,0,1,1,1)),
        ('all_wrong_split',(1,1,2,3,4)),('one_correct_split',(0,1,1,2,3)),('two_correct_distributed',(0,0,1,2,3))]:
        game=repair_game(classes)
        examples[name]={k:([str(x) for x in game[k]] if k!='distance' else game[k])
            for k in ('count','shapley','immediate','immediate_path','enabling','distance')}
    result=dict(identity='ORACLE_REPAIR_SHAPLEY_FIXED_STATE_DIAGNOSTIC_V1',
        manifest_sha256=manifest_hash,input_sha256=manifest['input_sha256'],
        exhaustive=exact,synthetic_examples=examples,historical=replay,
        verification=dict(network_attempts=__import__('sitecustomize').network_attempt_count(),
            preimport_network_guard=True,source_and_input_hashes_verified_before_after=True,
            prior_full_arithmetic_reused_and_crosschecked=31,
            real_api_calls=0,validation_raw_reads=0,test_raw_reads=0,historical_writes=0,
            runtime_method_changed=False),
        scientific_status='RETROSPECTIVE_DIAGNOSTIC_EFFICACY_NOT_ESTABLISHED')
    if result['verification']['network_attempts']:raise ValueError('NETWORK_ATTEMPT')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,ensure_ascii=True,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(exhaustive=exact,counters=replay['counters'],comparison=replay['selection_comparison']),allow_nan=False))


if __name__=='__main__':main()
