"""Post-execution integrity and paired diagnosis; never controls generation."""
from collections import Counter
from dataclasses import asdict
import hashlib
import itertools
import json
from pathlib import Path
import random

from .math_baseline_calibration import (digest,native_extract,primary_correctness,
    diagnostic_extract,grade_diagnostic,paired_comparison,DiagnosticExtraction,TRUNCATED)
from .math_calibration_accounting import read_ledger,CalibrationAbort
from .math_calibration_execution import PLAN_PATH,request_for,verify_bundle,sha,ATTEMPT
from ..persistence.durable_io import read_json,atomic_write_json
from ..persistence.provider_receipts import ProviderResponseReceipts
from ..governance.token_accounting import serialized_request


def bootstrap_paired(deltas,seed=81,resamples=10000):
    if not deltas:return None
    rng=random.Random(seed);n=len(deltas)
    values=sorted(sum(deltas[rng.randrange(n)] for _ in range(n))/n for _ in range(resamples))
    return dict(lower=values[int(.025*(resamples-1))],upper=values[int(.975*(resamples-1))],
        seed=seed,resamples=resamples,unit='PAIRED_EXAMPLE',interpretation='DESCRIPTIVE_DEVELOPMENT_UNCERTAINTY')


def summarize_scored(logicals,accounting,complete):
    arms={}
    for arm in ('A','B','C'):
        rows=[l for l in logicals if l['arm']==arm];n=len(rows)
        flags=('first_valid','first_correct','recovered_valid','recovered_correct')
        counts={field:sum(r[field] for r in rows) for field in flags}
        input_tokens=sum(r['input_tokens'] for r in rows);output_tokens=sum(r['output_tokens'] for r in rows)
        charged=sum(r['charged_tokens'] for r in rows)
        arms[arm]=dict(completed_logicals=n,**{f+'_count':v for f,v in counts.items()},
            **{f+'_rate':v/n if n else None for f,v in counts.items()},
            semantic_draws=sum(r['semantic_draws'] for r in rows),transport_retries=sum(r['transport_retries'] for r in rows),
            input_tokens=input_tokens,output_tokens=output_tokens,reported_tokens=input_tokens+output_tokens,
            charged_tokens=charged,average_charged_tokens=charged/n if n else None,
            charged_tokens_per_native_correct=charged/counts['recovered_correct'] if counts['recovered_correct'] else None,
            first_diagnostic_status=dict(Counter(r['first_diagnostic_status'] for r in rows)),
            final_diagnostic_status=dict(Counter(r['final_diagnostic_status'] for r in rows)),
            first_diagnostic_correct_count=sum(r['first_diagnostic_correct'] is True for r in rows),
            final_diagnostic_correct_count=sum(r['final_diagnostic_correct'] is True for r in rows),
            truncated_draws=sum(r['truncated_draws'] for r in rows),capacity_expanded_draws=sum(r['capacity_expanded_draws'] for r in rows),
            terminal_native_reasons=dict(Counter(r['terminal_native_reason'] or 'VALID' for r in rows)),
            attempt_native_reasons=dict(Counter(reason for r in rows for reason in r['attempt_native_reasons'])),
            subject_level={s+'|'+level:dict(n=sum(r['subject']==s and r['level']==level for r in rows),
                first_correct=sum(r['subject']==s and r['level']==level and r['first_correct'] for r in rows),
                recovered_correct=sum(r['subject']==s and r['level']==level and r['recovered_correct'] for r in rows))
                for s,level in sorted({(r['subject'],r['level']) for r in rows})})
    comparisons={}
    if complete:
        for left,right in itertools.combinations(('A','B','C'),2):
            lhs={r['example_id_sha256']:r for r in logicals if r['arm']==left}
            rhs={r['example_id_sha256']:r for r in logicals if r['arm']==right}
            paired=paired_comparison(lhs,rhs)
            for field,metric in paired['metrics'].items():
                metric['descriptive_bootstrap_95']=bootstrap_paired([int(rhs[k][field])-int(lhs[k][field]) for k in sorted(lhs)])
            paired['contrast']='PROMPT_ONLY' if (left,right)==('A','B') else 'OUTPUT_INTERFACE_WITH_DIFFERENT_PARSER'
            for field in ('first_diagnostic_correct','final_diagnostic_correct'):
                # Unknown/unparseable means no verified correct result on the
                # all-example denominator; unknown remains separately reported.
                improved=sum(lhs[k][field] is not True and rhs[k][field] is True for k in lhs)
                regressed=sum(lhs[k][field] is True and rhs[k][field] is not True for k in lhs)
                paired[field]=dict(improved=improved,regressed=regressed,paired_delta=(improved-regressed)/len(lhs))
            comparisons[left+'_'+right]=paired
    return dict(status='COMPLETE_PAIRED_DEVELOPMENT_DIAGNOSIS' if complete else 'PARTIAL_OPERATIONAL_FACTS_ONLY',
        efficacy_comparisons_permitted=complete,arms=arms,paired_comparisons=comparisons,
        accounting={k:v for k,v in accounting.items() if k!='inflight'},
        validation_calls=0,test_calls=0,optimizer_calls=0,stage2_executed=False,
        common_diagnostic_is_secondary=True,generalization_or_population_significance_claim=False)


def audit_results(root,prep,run_root):
    root=Path(root);run_root=Path(run_root)
    scope,plan,selected=verify_bundle(root,prep)
    summary=read_json(run_root/'execution_summary.json');accounting=read_ledger(run_root/'accounting')
    if (accounting['terminal']!=summary['status'] or accounting['reserved_total']!=0
            or accounting!=summary['accounting'] or accounting['scope_sha256']!=scope['startup_identity_sha256']):
        raise CalibrationAbort('RESULT_LEDGER_TERMINAL_MISMATCH')
    checkpoint=read_json(run_root/'checkpoint_private.json') if (run_root/'checkpoint_private.json').exists() else dict(
        scope_sha256=scope['startup_identity_sha256'],completed=[])
    if 'integrity_sha256' in checkpoint:
        seal=checkpoint.pop('integrity_sha256')
        if digest(checkpoint)!=seal:raise CalibrationAbort('RESULT_CHECKPOINT_SEAL_MISMATCH')
    if checkpoint['scope_sha256']!=scope['startup_identity_sha256'] or len(checkpoint['completed'])!=summary['completed_logicals']:
        raise CalibrationAbort('RESULT_CHECKPOINT_SCOPE_MISMATCH')
    events=[json.loads(line) for line in (run_root/'accounting/events.jsonl').read_bytes().splitlines()]
    reservations={e['reservation_id']:e for e in events if e['kind']=='RESERVE'}
    charges={e['reservation_id']:e for e in events if e['kind']=='CHARGE'}
    receipts=ProviderResponseReceipts(run_root/'provider_receipts_private',attempt_id=ATTEMPT,startup_identity_sha256=scope['startup_identity_sha256'])
    ledger_by_logical=Counter();checked_receipts=0
    for key,event in reservations.items():
        record=receipts.read(key);charge=charges[key]
        if (event['lane']!=record['lane'] or hashlib.sha256(serialized_request(record['request'])).hexdigest()!=event['wire_sha256']
                or charge['response_receipt'] is None or charge['response_receipt']['integrity_sha256']!=read_json(run_root/'provider_receipts_private'/ (key+'.json'))['integrity_sha256']):
            raise CalibrationAbort('RESULT_RECEIPT_LEDGER_WIRE_MISMATCH')
        lane=event['lane'];ledger_by_logical[(lane['arm'],lane['example_id_sha256'])]+=charge['charged'];checked_receipts+=1
    expected_order=[]
    for index,row in enumerate(selected):
        order=('A','B','C')[index%3:]+('A','B','C')[:index%3]
        expected_order.extend((arm,row['stable_example_id']) for arm in order)
    if [(l['arm'],l['example_id']) for l in checkpoint['completed']]!=expected_order[:len(checkpoint['completed'])]:
        raise CalibrationAbort('RESULT_DISPATCH_ORDER_MISMATCH')
    selected_by_id={r['stable_example_id']:r for r in selected};scored=[]
    for logical in checkpoint['completed']:
        row=selected_by_id[logical['example_id']];arm=logical['arm'];attempts=logical['attempts'];previous=None
        diagnostics=[];primary=[]
        if not 1<=len(attempts)<=4:raise CalibrationAbort('RESULT_SEMANTIC_COUNT_MISMATCH')
        for ordinal,attempt in enumerate(attempts,1):
            response=attempt['response'];native=native_extract(arm,response['text'],response['finish_reason'])
            if (native!=attempt['native'] or attempt['request']!=request_for(arm,row['content']['problem'],6144 if previous in TRUNCATED else 3600)
                    or attempt['lane']['semantic_draw']!=ordinal
                    or receipts.read(attempt['reservation_id'])['response']!=response
                    or ordinal<len(attempts) and native['valid']):
                raise CalibrationAbort('RESULT_DRAW_REPLAY_MISMATCH')
            # Extraction is immutable and completed before reference comparison.
            extraction=diagnostic_extract(response['text'],response['finish_reason'])
            diagnostics.append((extraction,grade_diagnostic(extraction,row['reference_final_answer'])))
            primary.append(primary_correctness(native,row['reference_final_answer']))
            previous=response['finish_reason']
        if not attempts[-1]['native']['valid'] and len(attempts)!=4:
            raise CalibrationAbort('RESULT_INCOMPLETE_LOGICAL_PRESENTED_AS_COMPLETE')
        if logical['reference']!=row['reference_final_answer']:
            raise CalibrationAbort('RESULT_REFERENCE_IDENTITY_MISMATCH')
        if (logical['first_native']!=attempts[0]['native'] or logical['recovered_native']!=attempts[-1]['native']
                or logical['semantic_draws']!=len(attempts)
                or logical['input_tokens']!=sum(a['response']['input_tokens'] for a in attempts)
                or logical['output_tokens']!=sum(a['response']['output_tokens'] for a in attempts)):
            raise CalibrationAbort('RESULT_LOGICAL_SUMMARY_MISMATCH')
        scored.append(dict(example_id_sha256=logical['example_id_sha256'],arm=arm,
            first_valid=attempts[0]['native']['valid'],first_correct=primary[0],
            recovered_valid=attempts[-1]['native']['valid'],recovered_correct=primary[-1],
            first_diagnostic_status=diagnostics[0][0].status,first_diagnostic_correct=diagnostics[0][1],
            final_diagnostic_status=diagnostics[-1][0].status,final_diagnostic_correct=diagnostics[-1][1],
            subject=logical['subject'],level=logical['level'],semantic_draws=len(attempts),transport_retries=logical['transport_retries'],
            input_tokens=logical['input_tokens'],output_tokens=logical['output_tokens'],
            charged_tokens=ledger_by_logical[(arm,logical['example_id_sha256'])],
            truncated_draws=sum(a['response']['finish_reason'] in TRUNCATED for a in attempts),
            capacity_expanded_draws=sum(a['output_capacity']==6144 for a in attempts),
            terminal_native_reason=attempts[-1]['native']['invalid_reason'],
            attempt_native_reasons=[a['native']['invalid_reason'] or 'VALID' for a in attempts]))
    complete=summary['status']=='EXECUTION_COMPLETE' and len(scored)==180
    if summary['status']=='EXECUTION_COMPLETE' and not complete:raise CalibrationAbort('RESULT_COMPLETE_PANEL_MISMATCH')
    result=summarize_scored(scored,accounting,complete)
    result['integrity']=dict(status='PASS',immutable_receipts_checked=checked_receipts,
        protocol_sha256=plan['protocol_sha256'],membership_sha256=plan['data']['membership_sha256'],
        source_sha=scope['source_sha'],startup_identity_sha256=scope['startup_identity_sha256'],
        checkpoint_sha256=sha(run_root/'checkpoint_private.json') if (run_root/'checkpoint_private.json').exists() else None,
        execution_summary_sha256=sha(run_root/'execution_summary.json'),secondary_scoring_never_written_to_primary=True)
    atomic_write_json(run_root/'scored_logicals_private.json',scored)
    atomic_write_json(run_root/'audited_summary_sanitized.json',result)
    return result
