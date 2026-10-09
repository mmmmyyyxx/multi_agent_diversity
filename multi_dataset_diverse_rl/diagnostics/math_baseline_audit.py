"""Read existing receipts and Optimize rows; write only new audit artifacts."""
from collections import Counter,defaultdict
from dataclasses import asdict
import hashlib,json
from pathlib import Path

from .math_baseline_calibration import (digest,diagnostic_extract,grade_diagnostic,
    primary_reason,secondary_flags,summarize_rounds,native_extract,primary_correctness)
from ..benchmarks.math_structured_answer import prediction_from_persisted,classify_prediction
from ..benchmarks.experiment_splits import ExperimentSplitReader,subject
from ..benchmarks.data_freeze import file_hash
from ..persistence.provider_receipts import digest as receipt_digest
from ..governance.token_accounting import serialized_request
from ..search.system_prompt import SystemPrompt,solver_messages

ATTEMPT='a4_v25_seed81_canary_attempt1'
BASE_SOURCE='9bbe1b502a3c8c6d43b2da959b8b8b0f2256b985'


def json_read(path):return json.loads(Path(path).read_bytes())


def optimize_rows(root,contract):
    reader=ExperimentSplitReader(root/contract['canonical_root'],root/contract['split_directory'],'math',
        expected_manifest_sha256=contract['split_manifest_sha256'],expected_protocol=contract['split_version'])
    # This guarded reader hashes source bytes but parses ONLY frozen Optimize
    # source indices. Validation, Test and Shadow content are never requested.
    from ..benchmarks.access import DataPurpose
    rows=reader.rows('optimize',DataPurpose.SEARCH_EVALUATION)
    return rows,reader


def cache_read(directory,key):
    row=json_read(directory/(key+'.json'))
    payload={k:v for k,v in row.items() if k!='integrity_seal'}
    if (digest(payload)!=row.get('integrity_seal') or digest(row['result'])!=row['response_sha256']
            or row['context']!=json_read(directory/'scope.json') or row['request_sha256']!=key):
        raise ValueError('AUDIT_CACHE_INTEGRITY_FAILURE')
    result=row['result'];prediction=prediction_from_persisted(result['resolved_prediction'])
    if len(result['original_realizations'])!=prediction.semantic_attempt_count:
        raise ValueError('AUDIT_RETRY_COUNT_MISMATCH')
    return result


def receipt_rows(directory):
    rows=[]
    for path in directory.glob('*.json'):
        row=json_read(path);payload={k:v for k,v in row.items() if k!='integrity_sha256'}
        if receipt_digest(payload)!=row['integrity_sha256'] or row['attempt_id']!=ATTEMPT:
            raise ValueError('AUDIT_RECEIPT_INTEGRITY_FAILURE')
        rows.append((row['reservation_id'],payload['record'],row['integrity_sha256']))
    return rows


def verify_ledger_events(rows):
    """Check the immutable journal without acquiring a writing ledger owner."""
    previous=None
    for sequence,row in enumerate(rows):
        payload={k:v for k,v in row.items() if k!='event_sha256'}
        if (digest(payload)!=row.get('event_sha256') or row.get('sequence')!=sequence
                or row.get('previous_sha256')!=previous):
            raise ValueError('AUDIT_LEDGER_HASH_CHAIN_FAILURE')
        previous=row['event_sha256']
    return dict(events=len(rows),last_event_sha256=previous,hash_chain_verified=True)


def validate_initial_panel(profiles):
    if len(profiles)!=5 or any(len(member)!=12 for member in profiles):
        raise ValueError('INITIAL_PANEL_SHAPE_MISMATCH')
    baseline=None
    for member,rows in enumerate(profiles):
        sources=[p['solver_trajectory']['source'] for p in rows]
        ids=[s['example_id'] for s in sources]
        if (len(set(ids))!=12 or any(s['member_id']!=member for s in sources)
                or baseline is not None and ids!=baseline):
            raise ValueError('INITIAL_PANEL_MEMBERSHIP_MISMATCH')
        baseline=ids


def verify_receipt_wire_join(reservations,records,execution_phase):
    for reservation,r,_ in records:
        event=reservations[reservation]
        # The accounting stage is the whole execution phase (Canary), whereas
        # receipt stage is the scientific substage (initial, shadow, etc.).
        if (event['request_sha256']!=hashlib.sha256(serialized_request(r['request'])).hexdigest()
                or event['stage']!=execution_phase or event['role']!=r['role']
                or event['model']!=r['request']['model'] or event['attempt_id']!=ATTEMPT):
            raise ValueError('AUDIT_LEDGER_RECEIPT_REQUEST_MISMATCH')


def deployment_audit(contract,records):
    requests=[r for _,r,_ in records if r['role']=='solver']
    expected=dict(model='qwen3-8b',temperature=0.2,top_p=0.8,presence_penalty=0,frequency_penalty=0,
        extra_body=dict(enable_thinking=False,top_k=20,min_p=0))
    checks=Counter();models=Counter();capacities=Counter();reasons=Counter();reasoning=Counter();wire_hashes=[]
    for r in requests:
        req=r['request'];capacities[str(req.get('max_tokens'))]+=1
        checks['request_parameters_match']+=int(all(req.get(k)==v for k,v in expected.items())
            and req.get('max_tokens') in (3600,6144) and type(req['extra_body'].get('enable_thinking')) is bool)
        wire=serialized_request(req);body=json.loads(wire)
        wire_hashes.append(hashlib.sha256(wire).hexdigest())
        checks['serialized_thinking_false']+=int(body.get('enable_thinking') is False)
        response=r.get('response')
        if response is None:continue
        raw=response.get('provider_http_response_body')
        if not isinstance(raw,dict):
            checks['raw_body_missing']+=1;continue
        choice=raw['choices'][0];message=choice['message']
        checks['content_equal']+=int(message.get('content')==response['text'])
        checks['finish_equal']+=int(choice.get('finish_reason')==response['finish_reason'])
        checks['usage_equal']+=int(raw['usage'].get('prompt_tokens')==response['input_tokens']
            and raw['usage'].get('completion_tokens')==response['output_tokens'])
        models[str(raw.get('model'))]+=1;reasons[str(choice.get('finish_reason'))]+=1
        reasoning['field_present']+=int('reasoning_content' in message)
        reasoning['nonempty']+=int(bool(message.get('reasoning_content')))
    successful=sum('response' in r for r in requests)
    matrix=dict(configuration_thinking_false='PASS' if contract['solver_decoding_policy']['enable_thinking'] is False else 'FAIL',
        dispatched_thinking_false='PASS' if checks['serialized_thinking_false']==len(requests) else 'FAIL',
        provider_returned_model_alias='PASS' if models==Counter({'qwen3-8b':successful}) else 'UNKNOWN',
        complete_assistant_content_saved='PASS' if checks['content_equal']==successful else 'FAIL',
        parser_input_equals_ordinary_content='UNKNOWN_PENDING_PROFILE_CACHE_JOIN',
        finish_reason_preserved='PASS' if checks['finish_equal']==successful else 'FAIL',
        retry_sampling_preserved='PASS' if checks['request_parameters_match']==len(requests) else 'FAIL',
        actual_weight_revision='UNKNOWN')
    return dict(matrix=matrix,frozen_configuration=contract['solver_decoding_policy'],
        all_solver_transport_requests=len(requests),all_solver_successes=successful,
        checks=dict(checks),response_model_alias_counts=dict(models),capacity_counts=dict(capacities),
        finish_reason_counts=dict(reasons),reasoning_field_observations=dict(reasoning),
        wire_body_identity_set_sha256=digest(sorted(wire_hashes)),
        verification_scope='SEALED_REQUEST_RECORDS_PLUS_FROZEN_SERIALIZER_AND_SEND_PATH_NOT_SERVER_WEIGHT_ATTESTATION',
        wire_extra_body='SDK_EXTRA_BODY_FLATTENED_TO_HTTP_JSON_TOP_LEVEL',
        provider_weight_revision_metadata='NOT_EXPOSED_IN_EXISTING_HTTP_BODIES',
        provider_honors_thinking_control='UNKNOWN_SERVER_INTERNAL_BEHAVIOR_NOT_ATTESTED',
        actual_backend_weights='UNKNOWN_NO_REVISION_ATTESTATION_NO_NEW_MODEL_PROBE')


def audit_initial(root):
    root=Path(root);run=root/'runs'/ATTEMPT/'execution'
    contract=json_read(root/f'experiments/execution_bindings/{ATTEMPT}.json')
    initial=json_read(run/'initial_state_private.json')
    recovered=json_read(root/'runs/v25_canary_terminal_recovery_20261009/initial_state_private.json')
    if initial!=recovered:raise ValueError('INITIAL_RECOVERY_STATE_DIFFERENT')
    rows,reader=optimize_rows(root,contract)
    examples={r['stable_example_id']:r for r in rows}
    records=receipt_rows(run/'provider_response_receipts_private')
    trace=[json.loads(line) for line in (run/'provider_trace_private.jsonl').read_text(encoding='utf-8').splitlines()]
    successes={(r['request_sha256'],r['semantic_attempt_no'],r['response']['response_id']):
        (reservation,r,seal) for reservation,r,seal in records if r['role']=='solver' and 'response' in r}
    trace_lookup={(r['request_sha256'],r['semantic_attempt_no'],r['response']['response_id']):r
        for r in trace if r['role']=='solver' and 'response' in r}
    # Error costs stay distinct from provider-reported successful usage.
    event_rows=[json.loads(line) for line in (root/'runs'/ATTEMPT/'accounting/events.jsonl').read_text(encoding='utf-8').splitlines()]
    ledger_integrity=verify_ledger_events(event_rows)
    reservations={e['reservation_id']:e for e in event_rows if e.get('kind')=='RESERVE'}
    verify_receipt_wire_join(reservations,records,contract['execution_phase'])
    validate_initial_panel(initial['diagnostics']['raw_profiles'])
    deployment=deployment_audit(contract,records)
    attempts=[];logicals=[];seen=set();profile_content_checks=0
    for member,profiles in enumerate(initial['diagnostics']['raw_profiles']):
        for profile in profiles:
            source=profile['solver_trajectory']['source'];xid=source['example_id'];key=source['request_sha256']
            if (member,xid) in seen:raise ValueError('INITIAL_LOGICAL_DUPLICATE')
            seen.add((member,xid))
            if source['split']!='optimize' or xid not in examples:raise ValueError('NONOPTIMIZE_DIAGNOSTIC_ACCESS')
            example=examples[xid];reference=example['reference_final_answer']
            prompt=SystemPrompt.from_dict(initial['member_prompts'][member])
            result=cache_read(run/'resolved_output_cache',key)
            prediction=prediction_from_persisted(profile['prediction'])
            if result['resolved_prediction']!=profile['prediction']:raise ValueError('INITIAL_CACHE_PROFILE_DIFFERENT')
            if any(p.prediction_valid for p in prediction.original_predictions[:-1]):
                raise ValueError('RETRY_AFTER_VALID_OUTPUT')
            if not 1<=len(prediction.original_predictions)<=4:
                raise ValueError('INITIAL_RETRY_SEQUENCE_INCOMPLETE')
            if not prediction.prediction_valid and len(prediction.original_predictions)!=4:
                raise ValueError('INITIAL_INVALID_RECOVERY_INCOMPLETE')
            if profile['solver_trajectory']['observed_response']!=prediction.text:
                raise ValueError('INITIAL_TRAJECTORY_TEXT_DIFFERENT')
            profile_content_checks+=1;native_rows=[];prior=None
            for ordinal,(realization,persisted) in enumerate(zip(result['original_realizations'],prediction.original_predictions,strict=True),1):
                replay=classify_prediction(realization['text'],realization['finish_reason'])
                if replay!=persisted:raise ValueError('OFFICIAL_PARSER_REPLAY_DIFFERENT')
                physical_key=(realization['request_sha256'],ordinal,realization['response_id'])
                reservation,receipt,seal=successes[physical_key]
                recorded=trace_lookup[physical_key]
                if (any(recorded.get(k)!=v for k,v in receipt.items() if k!='response')
                        or any(recorded['response'].get(k)!=receipt['response'].get(k)
                            for k in ('text','finish_reason','input_tokens','output_tokens','provider_http_response_body'))):
                    raise ValueError('TRACE_RECEIPT_CONTENT_OR_REQUEST_DIFFERENT')
                if receipt['stage']!='initial':raise ValueError('INITIAL_RECORD_STAGE_MISMATCH')
                req=receipt['request'];expected_capacity=6144 if prior is not None and prior.finish_reason in {'length','max_tokens','max_output_tokens'} else 3600
                if (req['messages']!=solver_messages(prompt,example['content']['problem'])
                        or req['max_tokens']!=expected_capacity or realization['text']!=receipt['response']['text']
                        or realization['finish_reason']!=receipt['response']['finish_reason']
                        or any(realization[k]!=receipt['response'][k] for k in ('input_tokens','output_tokens'))):
                    raise ValueError('INITIAL_WIRE_OR_CAPACITY_MISMATCH')
                extraction=diagnostic_extract(realization['text'],realization['finish_reason'])
                diagnostic_correct=grade_diagnostic(extraction,reference)
                native=native_extract('A',realization['text'],realization['finish_reason'])
                correct=primary_correctness(native,reference)
                row=dict(example_id=xid,example_id_sha256=hashlib.sha256(xid.encode()).hexdigest(),member_id=member,
                    logical_request_sha256=key,prompt_sha256=source['mutable_prompt_sha256'],
                    semantic_attempt_index=ordinal,provider_request_sha256=realization['request_sha256'],
                    provider_receipt_sha256=seal,reservation_id=reservation,
                    finish_reason=realization['finish_reason'],input_tokens=realization['input_tokens'],
                    output_tokens=realization['output_tokens'],output_capacity=req['max_tokens'],
                    native_valid=replay.prediction_valid,native_invalid_reason=replay.invalid_reason,
                    official_correctness_under_frozen_rule=correct,
                    primary_reason=primary_reason(realization['text'],realization['finish_reason'],replay),
                    secondary_flags=secondary_flags(realization['text'],realization['finish_reason']),
                    diagnostic=asdict(extraction),diagnostic_correct=diagnostic_correct,
                    original_text=realization['text'],reference=reference,
                    source_attempt_id=ATTEMPT,source_startup_sha256=source.get('startup_identity_sha256') or
                        json_read(root/'runs'/ATTEMPT/'prep/prep.json')['startup_identity_sha256'])
                attempts.append(row);native_rows.append(row);prior=replay
            final=native_rows[-1];retry=profile['solver_trajectory']['retry_summary']
            logicals.append(dict(example_id=xid,member_id=member,logical_request_sha256=key,
                prompt_sha256=source['mutable_prompt_sha256'],semantic_draws=len(native_rows),
                native_valid=prediction.prediction_valid,recovered_valid=prediction.recovered_invalid,
                terminal_invalid=prediction.terminal_invalid,repeated_format_failure=retry['repeated_format_failure'],
                transport_retries=prediction.transport_retry_attempts,primary_reason=final['primary_reason'],
                official_correctness=final['official_correctness_under_frozen_rule'],
                final_diagnostic=final['diagnostic'],final_diagnostic_correct=final['diagnostic_correct'],
                diagnostic_correct_on_any_draw=any(a['diagnostic_correct'] is True and not a['diagnostic']['truncated'] for a in native_rows),
                observed_truncation=any(a['finish_reason'] in {'length','max_tokens','max_output_tokens'} for a in native_rows),
                input_tokens=sum(a['input_tokens'] for a in native_rows),output_tokens=sum(a['output_tokens'] for a in native_rows),
                subject=subject(example),level=example['content']['level']))
    initial_errors=[]
    for reservation,r,seal in records:
        if r['role']=='solver' and r['stage']=='initial' and 'response' not in r:
            # The immutable charge is joined by reservation ID, not response text.
            matching=[e for e in event_rows if str(e.get('reservation_id',e.get('key','')))==reservation]
            charges=[e for e in matching if e.get('kind')=='CHARGE']
            initial_errors.append(dict(reservation_id=reservation,request_sha256=r['request_sha256'],
                semantic_attempt_no=r['semantic_attempt_no'],physical_attempt_no=r['physical_attempt_no'],
                receipt_sha256=seal,error_category=r.get('error_category'),
                accounting_events=matching,conservative_charge=sum(e['charged'] for e in charges)))
    if len(logicals)!=60 or len(attempts)!=192 or len(initial_errors)!=2:
        raise ValueError('INITIAL_DENOMINATOR_DIFFERENT_FROM_PUBLISHED_REPORT')
    expected_scores=[sum(l['official_correctness'] for l in logicals if l['member_id']==m) for m in range(5)]
    if expected_scores!=initial['member_scores']:raise ValueError('OFFICIAL_INITIAL_SCORES_DIFFERENT')
    for row in attempts:
        logical=next(l for l in logicals if (l['member_id'],l['example_id'])==(row['member_id'],row['example_id']))
        row.update(final_recovered_valid=logical['recovered_valid'],final_native_valid=logical['native_valid'],
            final_terminal_invalid=logical['terminal_invalid'],repeated_format_failure=logical['repeated_format_failure'],
            logical_observed_truncation=logical['observed_truncation'],
            example_input_sha256=examples[row['example_id']]['input_sha256'],
            example_content_sha256=examples[row['example_id']]['content_sha256'],
            source_split_manifest_sha256=contract['split_manifest_sha256'])
    deployment['matrix']['parser_input_equals_ordinary_content']='PASS'
    deployment['initial_profile_lossless_content_checks']=profile_content_checks
    deployment['initial_attempt_lossless_cache_receipt_trace_checks']=len(attempts)
    deployment['initial_capacity_sequence_verified']=True
    deployment['all_provider_receipts_joined_to_exact_ledger_wire_identity']=len(records)
    deployment['initial_cache_usage_matches_immutable_receipts']=len(attempts)
    summary=dict(evidence_level='VERIFIED',logical_examples=len(logicals),successful_responses=len(attempts),
        transport_timeouts=len(initial_errors),official_final_valid=sum(l['native_valid'] for l in logicals),
        official_final_invalid=sum(l['terminal_invalid'] for l in logicals),
        official_member_correct_counts=expected_scores,official_member_denominators=[12]*5,
        official_vote_correct=initial['team_scores']['vote_correct_count'],
        official_oracle_correct=sum(any(l['official_correctness'] for l in logicals if l['example_id']==xid) for xid in {l['example_id'] for l in logicals}),
        official_accuracy_unchanged=True,offline_diagnostic_scores_never_written_back=True,
        repeated_native_reason_failures=sum(l['repeated_format_failure'] for l in logicals),
        native_attempt_reasons=dict(Counter(a['native_invalid_reason'] or 'VALID' for a in attempts)),
        attempt_primary_reasons=dict(Counter(a['primary_reason'] for a in attempts)),
        logical_terminal_primary_reasons=dict(Counter(l['primary_reason'] for l in logicals)),
        invalid_logical_terminal_primary_reasons=dict(Counter(l['primary_reason'] for l in logicals if l['terminal_invalid'])),
        final_invalid_diagnostic_status=dict(Counter(l['final_diagnostic']['status'] for l in logicals if l['terminal_invalid'])),
        final_invalid_explicit_parseable=sum(l['terminal_invalid'] and l['final_diagnostic']['status']=='EXPLICIT_PARSEABLE_RESULT' for l in logicals),
        final_invalid_diagnostic_correct=sum(l['terminal_invalid'] and l['final_diagnostic_correct'] is True and not l['final_diagnostic']['truncated'] for l in logicals),
        invalid_logicals_with_correct_explicit_draw=sum(l['terminal_invalid'] and l['diagnostic_correct_on_any_draw'] for l in logicals),
        rounds=summarize_rounds(logicals,attempts,initial_errors),
        reported_success_tokens=sum(a['input_tokens']+a['output_tokens'] for a in attempts),
        extra_successful_semantic_draws=sum(l['semantic_draws']-1 for l in logicals),
        extra_format_draw_reported_tokens=sum(a['input_tokens']+a['output_tokens'] for a in attempts if a['semantic_attempt_index']>1),
        repeated_failure_reported_tokens=sum(l['input_tokens']+l['output_tokens'] for l in logicals if l['repeated_format_failure']),
        conservative_initial_timeout_charge=sum(e['conservative_charge'] for e in initial_errors),
        capacity_triggered_draws=sum(a['output_capacity']==6144 for a in attempts),
        truncated_attempts=sum(a['finish_reason'] in {'length','max_tokens','max_output_tokens'} for a in attempts),
        diagnostic_is_not_new_baseline=True,api_calls=0,validation_rows_parsed=0,test_rows_parsed=0)
    summary['ledger_integrity']=ledger_integrity
    summary['first_draw_primary_reasons']=dict(Counter(a['primary_reason'] for a in attempts if a['semantic_attempt_index']==1))
    summary['reported_tokens_by_attempt_primary_reason']={reason:sum(a['input_tokens']+a['output_tokens'] for a in attempts if a['primary_reason']==reason) for reason in sorted({a['primary_reason'] for a in attempts})}
    summary['truncated_draw_reported_tokens']=sum(a['input_tokens']+a['output_tokens'] for a in attempts if a['finish_reason'] in {'length','max_tokens','max_output_tokens'})
    summary['capacity_expanded_draw_reported_tokens']=sum(a['input_tokens']+a['output_tokens'] for a in attempts if a['output_capacity']==6144)
    summary['retry_tokens_for_repeated_native_failures']=sum(a['input_tokens']+a['output_tokens'] for a in attempts if a['semantic_attempt_index']>1 and next(l['repeated_format_failure'] for l in logicals if l['member_id']==a['member_id'] and l['example_id']==a['example_id']))
    return dict(summary=summary,deployment=deployment,attempts=attempts,logicals=logicals,
        errors=initial_errors,optimize_rows=rows,reader=reader)
