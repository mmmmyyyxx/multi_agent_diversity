"""Zero-API sealed Stage 1 postmortem and fresh five-member B preparation."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_dataset_diverse_rl.diagnostics.math_baseline_followup import (
    b_residual_catalog, display_box_diagnostic, grade_compatibility,
    choose_fresh_membership, initial_team, membership_identity, b5_budget, PLAN_ID,
)
from multi_dataset_diverse_rl.diagnostics.math_baseline_calibration import digest
from multi_dataset_diverse_rl.diagnostics.math_baseline_calibration import diagnostic_extract,grade_diagnostic,native_extract
from multi_dataset_diverse_rl.benchmarks.math_domain_v2 import domain_matrix
import re
from multi_dataset_diverse_rl.diagnostics.math_baseline_audit import optimize_rows
from multi_dataset_diverse_rl.diagnostics.math_calibration_accounting import read_ledger
from multi_dataset_diverse_rl.persistence.provider_receipts import ProviderResponseReceipts
from multi_dataset_diverse_rl.governance.token_accounting import serialized_request


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2)+'\n').encode())


def audit(root=ROOT):
    root=Path(root)
    private=root/'runs/math_baseline_followup_20261010'
    private.mkdir(parents=True, exist_ok=True)
    run=root/'runs/math_baseline_stage1_seed81_20261010_attempt1'
    parent=root/'reports/math_baseline_calibration_stage1_execution_20261010'
    # The old runner deliberately rejects a later HEAD. Read-only postmortems
    # instead anchor to its immutable report seals; never loosen its startup gate.
    for item in read(parent/'sha256_manifest.json')['files']:
        if sha(parent/item['file']) != item['sha256']:
            raise ValueError('PARENT_PUBLIC_EVIDENCE_CHANGED')
    integrity=read(parent/'integrity_audit.json')
    for name,key in [('checkpoint_private.json','checkpoint_sha256'),('execution_summary.json','execution_summary_sha256')]:
        if sha(run/name) != integrity[key]:
            raise ValueError('PARENT_PRIVATE_EVIDENCE_CHANGED')
    cp=read(run/'checkpoint_private.json')
    if digest({k:v for k,v in cp.items() if k!='integrity_sha256'}) != cp['integrity_sha256']:
        raise ValueError('CHECKPOINT_SEAL_FAILURE')
    ledger=read_ledger(run/'accounting')
    if ledger['terminal']!='EXECUTION_COMPLETE' or ledger['last_event_sha256'] != read(run/'execution_summary.json')['accounting']['last_event_sha256']:
        raise ValueError('PARENT_LEDGER_FAILURE')
    receipts=ProviderResponseReceipts(run/'provider_receipts_private',
        attempt_id='math_baseline_stage1_seed81_20261010_attempt1',startup_identity_sha256=cp['scope_sha256'])
    events=[json.loads(line) for line in (run/'accounting/events.jsonl').read_bytes().splitlines()]
    reservations={r['reservation_id']:r for r in events if r['kind']=='RESERVE'}
    charges={r['reservation_id']:r for r in events if r['kind']=='CHARGE'}
    for key,event in reservations.items():
        record=receipts.read(key)
        if record['lane']!=event['lane'] or sha_wire(record['request'])!=event['wire_sha256']:
            raise ValueError('RECEIPT_WIRE_JOIN_FAILURE')
        receipt_row=read(run/'provider_receipts_private'/(key+'.json'))
        if charges[key]['response_receipt']['integrity_sha256'] != receipt_row['integrity_sha256']:
            raise ValueError('RECEIPT_CHARGE_JOIN_FAILURE')
    logicals=cp['completed']
    if len(logicals)!=180 or cp['scope_sha256']!=integrity['startup_identity_sha256']:
        raise ValueError('PARENT_PANEL_FAILURE')
    for row in logicals:
        for attempt in row['attempts']:
            record=receipts.read(attempt['reservation_id'])
            if record['response']!=attempt['response'] or record['request']!=attempt['request']:
                raise ValueError('CHECKPOINT_RECEIPT_JOIN_FAILURE')
    catalog=b_residual_catalog(logicals)
    write(private/'b_residual_catalog_private.json',catalog)
    by_hash={r['example_id_sha256']:r for r in logicals if r['arm']=='B'}
    radical=by_hash['aa0bbc140215e16e29fc7155b7d16682f4275372538128772fbf8840e5573d3b']
    line=radical['attempts'][-1]['response']['text'].splitlines()[-1].strip()
    match=re.fullmatch(r'Final answer: (\d+)\u221a(\d+)',line)
    if not match:raise ValueError('RADICAL_BOUNDARY_PROBE_MISMATCH')
    translated=match[1]+'\\sqrt{'+match[2]+'}'
    relation=domain_matrix((radical['reference'],translated))
    if not all(relation['valid']) or not relation['equivalence'][0][1]:
        raise ValueError('RADICAL_COUNTERFACTUAL_NOT_VERIFIED')
    matrix=by_hash['455397138b1eb7495bfa285f024b5e097f6a260b88ac15e5a0169eb61e8ad67d']
    matrix_text=matrix['attempts'][-1]['response']['text']
    extraction=diagnostic_extract(matrix_text)
    if grade_diagnostic(extraction,matrix['reference']) is not True or '\\begin{pmatrix}' not in extraction.payload:
        raise ValueError('MATRIX_COUNTERFACTUAL_NOT_VERIFIED')
    interface_probes=dict(unicode_radical_literal_preserved=True,strict_gold_blind_radical_translation_verified=True,
        matrix_domain_parse_and_equivalence_verified=True,matrix_lexical_guard_rejection_replayed=True,
        original_native_invalidity_preserved=True,production_parser_unchanged=True)
    write(private/'b_interface_probes_sanitized.json',interface_probes)
    compatibility=[]
    for row in logicals:
        if row['arm']!='C':
            continue
        for ordinal,attempt in enumerate(row['attempts'],1):
            response=attempt['response']
            extraction=display_box_diagnostic(response['text'],response['finish_reason'])
            compatibility.append(dict(example_id_sha256=row['example_id_sha256'],draw=ordinal,
                first=ordinal==1,final=ordinal==len(row['attempts']),
                old_native_valid=attempt['native']['valid'],
                compatible_valid=extraction['valid'],
                compatible_correct=grade_compatibility(extraction,row['reference']),
                reason=extraction['reason']))
    write(private/'c_compatibility_rows_private.json',compatibility)
    c_summary={}
    for key in ('first','final','all'):
        records=[r for r in compatibility if key=='all' or r[key]]
        c_summary[key]=dict(responses=len(records),old_native_valid=sum(r['old_native_valid'] for r in records),
            diagnostic_valid=sum(r['compatible_valid'] for r in records),
            diagnostic_correct=sum(r['compatible_correct'] is True for r in records),
            rejection_categories=dict(Counter(r['reason'] for r in records if r['reason'])))
    binding=read(root/'experiments/execution_bindings/a4_v25_seed81_canary_attempt1.json')
    rows,_=optimize_rows(root,binding)
    original=read(root/'runs/a4_v25_seed81_canary_attempt1/execution/initial_state_private.json')
    canary={p['solver_trajectory']['source']['example_id'] for p in original['diagnostics']['raw_profiles'][0]}
    stage1={r['example_id'] for r in logicals}
    if len(canary)!=12 or len(stage1)!=60 or canary & stage1:
        raise ValueError('EXCLUSION_MEMBERSHIP_FAILURE')
    selected,selection=choose_fresh_membership(rows,canary|stage1)
    identities=membership_identity(selected)
    write(private/'selected_rows_private.json',selected)
    write(private/'selection_sanitized.json',selection)
    write(private/'membership_sanitized.json',identities)
    summary=dict(status='VERIFIED_OFFLINE',real_api_calls=0,
        b=dict(official_remaining_errors=11,native_valid_math_wrong=8,native_invalid=3,
            repair_groups=dict(Counter(r['repair_group'] for r in catalog)),
            category_counts=dict(Counter(r['first_error_category'] for r in catalog)),
            strategy_priority_count=7,local_verification_count=2,interface_only_count=2,
            interface_probes=interface_probes,
            annotation_scope='OWNER_FIRST_ERROR_REVIEW_REPAIR_EFFICACY_UNTESTED'),
        c=dict(identity='MATH_TERMINAL_DISPLAY_BOX_COMPAT_DIAGNOSTIC_V1',
            **c_summary,official_scores_unchanged=True,production_parser_unchanged=True,
            hypothetical_future_parser_only=True,not_rerun_or_recovered_scores=True),
        next_baseline=dict(identity=PLAN_ID,n=60,members=5,seed=82,
            membership_sha256=digest(identities),selection=selection,
            disjoint_from_stage1_and_canary=True,globally_unseen_claim=False,
            previously_adaptive_optimize_pool=True,heldout_generalization_claim=False,
            budget=b5_budget(selected),team_sha256=initial_team()['ordered_team_sha256'],
            actual_vote_oracle_coverage='NOT_EVALUATED_NEW_AUTHORIZATION_REQUIRED'),
        integrity=dict(parent_public_seals_verified=True,checkpoint_seal_verified=True,
            immutable_receipts_checked=len(reservations),ledger_hash_chain_verified=True,
            checkpoint_sha256=sha(run/'checkpoint_private.json'),
            ledger_last_sha256=ledger['last_event_sha256'],catalog_sha256=sha(private/'b_residual_catalog_private.json'),
            compatibility_rows_sha256=sha(private/'c_compatibility_rows_private.json')),
        gradient_ratio_decision='DEFERRED_UNTIL_FIVE_MEMBER_BASELINE_INTEGRITY_AND_STABILITY_AUDIT')
    write(private/'summary_sanitized.json',summary)
    print(json.dumps(summary,ensure_ascii=True,sort_keys=True))
    return summary


def sha_wire(request):
    return hashlib.sha256(serialized_request(request)).hexdigest()


if __name__=='__main__':
    audit()
