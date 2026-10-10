"""Integrity-first B-five-member results. Called only after generation ends."""
from collections import Counter
import hashlib
import json
from pathlib import Path

from .math_baseline_calibration import digest,native_extract,TRUNCATED
from .math_baseline_followup import panel_metrics
from .math_b5_execution import verify_bundle,request_for,validate_response,ATTEMPT,POLICY,RETRYABLE
from .math_b5_accounting import read_ledger,CalibrationAbort
from ..governance.token_accounting import serialized_request,reliable_usage
from ..persistence.provider_receipts import ProviderResponseReceipts
from ..persistence.durable_io import read_json,atomic_write_json


def summarize_complete(logicals,accounting,charges):
    grouped={}
    for row in logicals:
        grouped.setdefault(row['example_id_sha256'],{})[row['member']]=row
    if len(logicals)!=5*len(grouped) or any(set(v)!=set(range(5)) for v in grouped.values()):
        raise CalibrationAbort('B5_COMPLETE_MEMBER_PANEL_REQUIRED')
    phases={};n=len(grouped)
    for name,field in [('first','first_native'),('recovered','recovered_native')]:
        scored=[panel_metrics([v[i][field] for i in range(5)],v[0]['reference']) for v in grouped.values()]
        member_valid=[sum(r['member_valid'][i] for r in scored) for i in range(5)]
        phases[name]=dict(examples=n,member_valid_counts=member_valid,
            member_correct_counts=[sum(r['member_correct'][i] for r in scored) for i in range(5)],
            vote_correct_count=sum(r['vote_correct'] for r in scored),
            oracle_correct_count=sum(r['oracle_correct'] for r in scored),
            oracle_minus_vote_count=sum(r['oracle_correct'] and not r['vote_correct'] for r in scored),
            correct_coverage_histogram={str(i):sum(r['correct_member_count']==i for r in scored) for i in range(6)},
            uniquely_covering_member_counts=[sum(r['unique_cover_member']==i for r in scored) for i in range(5)],
            vote_abstentions=dict(Counter(r['selection']['reason'] for r in scored if r['selection']['winner_member'] is None)),
            total_valid_count=sum(member_valid),logical_denominator=5*n)
    recovery_charge=sum(c['charged'] for c in charges if c['lane']['semantic_draw']>1)
    total=accounting['charged_total'];fraction=recovery_charge/total if total else None
    gate=POLICY['stability_gate']
    stable=bool(n and total and phases['first']['total_valid_count']/(5*n)>=gate['first_valid_min']
        and phases['recovered']['total_valid_count']/(5*n)>=gate['recovered_valid_min']
        and min(phases['recovered']['member_valid_counts'])/n>=gate['recovered_valid_member_min']
        and fraction is not None and fraction<=gate['recovery_charge_fraction_max'])
    return dict(status='COMPLETE_DEVELOPMENT_DIAGNOSIS',phases=phases,
        cost=dict(charged_total=total,format_recovery_charged_tokens=recovery_charge,
            recovery_charge_fraction=fraction,physical_attempts=accounting['physical_attempts'],
            semantic_draws=sum(r['semantic_draws'] for r in logicals),
            truncation_draws=sum(a['response']['finish_reason'] in TRUNCATED for r in logicals for a in r['attempts'])),
        format_stability_gate='PASSED' if stable else 'NOT_PASSED',format_stability_thresholds=gate,
        gradient_ratio_change='NOT_AUTHORIZED_SEPARATE_DESIGN_REQUIRED',
        generalization_claim=False,member_pseudoreplication=False)


def audit_results(root,prep):
    root=Path(root);scope,_,selected=verify_bundle(root,prep)
    run=root/scope['run_root'];summary=read_json(run/'execution_summary.json')
    accounting=read_ledger(run/'accounting')
    if (summary['accounting']!=accounting or accounting['reserved_total']
            or accounting['scope_sha256']!=scope['startup_identity_sha256']
            or accounting['terminal']!=summary['status']):
        raise CalibrationAbort('B5_RESULT_ACCOUNTING_MISMATCH')
    cp=read_json(run/'checkpoint_private.json') if (run/'checkpoint_private.json').exists() else None
    if cp is not None and (digest({k:v for k,v in cp.items() if k!='integrity_sha256'})!=cp['integrity_sha256']
            or cp['scope_sha256']!=scope['startup_identity_sha256']):
        raise CalibrationAbort('B5_CHECKPOINT_SEAL_FAILURE')
    logicals=cp['completed'] if cp else []
    expected_order=[(r['stable_example_id'],i) for ix,r in enumerate(selected)
        for i in tuple(range(5))[ix%5:]+tuple(range(5))[:ix%5]]
    if [(r['example_id'],r['member']) for r in logicals]!=expected_order[:len(logicals)]:
        raise CalibrationAbort('B5_DISPATCH_ORDER_MISMATCH')
    events=[json.loads(line) for line in (run/'accounting/events.jsonl').read_bytes().splitlines()]
    reserves={r['reservation_id']:r for r in events if r['kind']=='RESERVE'}
    charges={r['reservation_id']:r for r in events if r['kind']=='CHARGE'}
    receipts=ProviderResponseReceipts(run/'provider_receipts_private',attempt_id=ATTEMPT,
        startup_identity_sha256=scope['startup_identity_sha256'])
    joined=[]
    draws={}
    for key,reserve in reserves.items():
        record=receipts.read(key);receipt=read_json(run/'provider_receipts_private'/(key+'.json'))
        if (record['lane']!=reserve['lane']
                or hashlib.sha256(serialized_request(record['request'])).hexdigest()!=reserve['wire_sha256']
                or charges[key]['response_receipt']['integrity_sha256']!=receipt['integrity_sha256']):
            raise CalibrationAbort('B5_RECEIPT_LEDGER_JOIN_FAILURE')
        lane=reserve['lane']
        draws.setdefault((lane['example_id_sha256'],lane['member'],lane['semantic_draw']),[]).append((reserve,record))
        if 'response' in record:
            usage=record['response'];reliable=reliable_usage(usage,reserve['bound'])
            charge=usage['input_tokens']+usage['output_tokens'] if reliable else reserve['bound']['amount']
            if charges[key]['charged']!=charge or charges[key]['reliable_usage']!=reliable:
                raise CalibrationAbort('B5_USAGE_CHARGE_JOIN_FAILURE')
        joined.append(charges[key]|dict(lane=reserve['lane']))
    if len(logicals)!=summary['completed_logicals']:
        raise CalibrationAbort('B5_COMPLETED_COUNT_MISMATCH')
    if summary['status']!='EXECUTION_COMPLETE':
        result=dict(status='PARTIAL_OPERATIONAL_FACTS_ONLY',completed_logicals=len(logicals),
            accounting={k:v for k,v in accounting.items() if k!='inflight'},efficacy_comparisons_permitted=False)
    else:
        if len(logicals)!=300:
            raise CalibrationAbort('B5_FIXED_PANEL_INCOMPLETE')
        lookup={r['stable_example_id']:r for r in selected}
        successful_keys=set()
        for logical in logicals:
            row=lookup[logical['example_id']];previous=None;attempts=logical['attempts']
            if (not 1<=len(attempts)<=4 or logical['reference']!=row['reference_final_answer']
                    or logical['example_id_sha256']!=hashlib.sha256(logical['example_id'].encode()).hexdigest()
                    or logical['semantic_draws']!=len(attempts)
                    or logical['input_tokens']!=sum(a['response']['input_tokens'] for a in attempts)
                    or logical['output_tokens']!=sum(a['response']['output_tokens'] for a in attempts)):
                raise CalibrationAbort('B5_LOGICAL_CONTRACT_MISMATCH')
            physical_retries=0
            for ordinal,attempt in enumerate(attempts,1):
                key=attempt['reservation_id'];record=receipts.read(key);response=attempt['response'];lane=attempt['lane']
                native=native_extract('B',response['text'],response['finish_reason'])
                physical=draws[(logical['example_id_sha256'],logical['member'],ordinal)]
                if (len(physical)>21 or [r['lane']['physical_attempt'] for r,_ in physical]!=list(range(1,len(physical)+1))
                        or any('response' in record or record.get('error_category') not in RETRYABLE for _,record in physical[:-1])
                        or 'response' not in physical[-1][1]
                        or any(record['request']!=attempt['request'] for _,record in physical)):
                    raise CalibrationAbort('B5_TRANSPORT_RETRY_REPLAY_MISMATCH')
                physical_retries+=len(physical)-1
                if (key in successful_keys or native!=attempt['native'] or response!=record['response']
                        or attempt['request']!=request_for('B',row['content']['problem'],6144 if previous in TRUNCATED else 3600)
                        or record['request']!=attempt['request'] or record['lane']!=lane
                        or lane['semantic_draw']!=ordinal or lane['member']!=logical['member']
                        or lane['example_id_sha256']!=logical['example_id_sha256'] or lane['arm']!='B'
                        or lane['stage']!='b_five_member_development' or lane['attempt_id']!=ATTEMPT
                        or ordinal<len(attempts) and native['valid']
                        or attempt['charged_tokens']!=charges[key]['charged']):
                    raise CalibrationAbort('B5_SEMANTIC_REPLAY_MISMATCH')
                validate_response(response,attempt['request'])
                successful_keys.add(key);previous=response['finish_reason']
            if physical_retries!=logical['transport_retries']:
                raise CalibrationAbort('B5_TRANSPORT_RETRY_COUNT_MISMATCH')
            if (not attempts[-1]['native']['valid'] and len(attempts)!=4
                    or logical['first_native']!=attempts[0]['native'] or logical['recovered_native']!=attempts[-1]['native']):
                raise CalibrationAbort('B5_FINAL_NATIVE_MISMATCH')
        receipt_successes={key for key in reserves if 'response' in receipts.read(key)}
        if successful_keys!=receipt_successes or len(draws)!=len(successful_keys):
            raise CalibrationAbort('B5_UNEXPLAINED_SUCCESSFUL_RECEIPT')
        result=summarize_complete(logicals,accounting,joined)
        result['integrity']='VALID_FIXED_PANEL_SOURCE_LEDGER_RECEIPT_NATIVE_REPLAY'
    atomic_write_json(run/'audited_summary_sanitized.json',result)
    return result
