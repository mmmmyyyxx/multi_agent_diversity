"""Post-execution only: independent zero-API audit of a pre-cluster Pilot abort.

This module is promoted to governance after the frozen execution is terminal.
It never composes a search, creates a provider, or authorizes an execution.
"""
from collections import Counter,defaultdict
from dataclasses import asdict,replace
import csv
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace as NS

from .artifacts import scan_sanitized_artifacts,build_sha256_manifest
from .unified_execution import validate_frozen_source,verify_source_commit
from .token_accounting import TokenLedger,POLICY_40M
from ..benchmarks.math_domain_binding import execution_binding
from ..benchmarks.math_solver_decoding import generation_request_fields
from ..benchmarks.math_prediction_validity import classify_prediction,resolve_predictions
from ..search.scientific_aggregation import EquivalencePluralityAggregation
from ..search.binary_responsibility import BinaryPluralityResponsibilityAnalyzer,binary_plurality_snapshot,observation_from_outputs
from ..search.binary_runtime import BinaryEvidenceSource
from ..search.history import HistoryState
from ..search.policies import TargetPolicyV1
from ..search.variable_evidence import VariableEvidenceFeasibilityV1
from ..search.textual_gradients import validate_gradient,GRADIENT_POLICY
from ..search.pattern_primitives import single_failure_example
from ..search.schemas import SearchContractError
from ..search.provider_runtime import RequestBroker


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def lines(path):return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x] if Path(path).exists() else []
def text_hash(value):return sha256(value.encode()).hexdigest()
def digest(value):return text_hash(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True))
def hashes(root):return {p.relative_to(root).as_posix():sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob('*')) if p.is_file()}
def write(root,name,value):
    path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
    assert not path.exists(),name
    path.write_bytes((json.dumps(value,sort_keys=True,indent=2)+'\n').encode())


def reconstruct_predictions(trace,ledger):
    grouped=defaultdict(list);failures=Counter()
    for row in trace:
        if row['role']=='solver' and 'response' in row:grouped[row['request_sha256']].append(row)
    for row in ledger:
        if row['kind']=='FAILURE' and row['role']=='solver':failures[row['request_sha256']]+=1
    predictions={}
    for key,rows in grouped.items():
        assert [r['semantic_attempt_no'] for r in rows]==list(range(1,len(rows)+1))
        predictions[key]=resolve_predictions(tuple(classify_prediction(r['response']['text'],r['response']['finish_reason']) for r in rows),failures[key])
    validity=[r for r in ledger if r['kind']=='PREDICTION_VALIDITY']
    for r in validity:
        p=predictions[r['request_sha256']]
        assert sha256(p.identity_bytes()).hexdigest()==r['prediction_sha256']
        assert p.prediction_valid==r['prediction_valid'] and p.invalid_reason==r['invalid_reason']
    return predictions,validity


def measure(benchmark,examples,profiles):
    parsed=[[benchmark.parse_member_output(raw,e.item) for raw,e in zip(member,examples,strict=True)] for member in profiles]
    bits=[[bool(benchmark.score_member_output(p,e.reference)==1) for p,e in zip(member,examples,strict=True)] for member in parsed]
    votes=[bool(benchmark.score_member_output(EquivalencePluralityAggregation().aggregate_sync(item=e.item,
        member_outputs=tuple(profiles[m][j] for m in range(5)),benchmark=benchmark).parsed_output,e.reference)==1)
        for j,e in enumerate(examples)]
    n=len(examples);counts=[sum(b) for b in bits];vote=sum(votes);oracle=sum(any(bits[m][j] for m in range(5)) for j in range(n))
    return dict(count=n,member_correct_counts=counts,vote_correct_count=vote,oracle_correct_count=oracle,
        VoteAcc=vote/n,OracleAcc=oracle/n,MeanMemberAcc=sum(counts)/(5*n),MinMemberAcc=min(counts)/n,
        invalid_count_by_member=[sum(not p.valid for p in member) for member in parsed],
        mean_disagreement=sum(len({str(parsed[m][j].answer) for m in range(5) if parsed[m][j].valid}) for j in range(n))/n),bits


def snapshot_measure(snapshot,benchmark,examples):
    metrics,bits=measure(benchmark,examples,snapshot['diagnostics']['raw_profiles'])
    assert metrics['member_correct_counts']==snapshot['member_scores']
    assert metrics['vote_correct_count']==snapshot['team_scores']['vote_correct_count']
    assert [sum(b) for b in bits]==snapshot['member_scores']
    for j,state in enumerate(snapshot['diagnostics']['team_states']):
        assert state['team_correctness']==[bits[m][j] for m in range(5)]
    return metrics


def who_reconstruction(snapshot,benchmark,examples,history):
    profiles=snapshot['diagnostics']['raw_profiles']
    obs=tuple(observation_from_outputs(benchmark=benchmark,item=e.item,
        member_outputs=tuple(profiles[m][j] for m in range(5)),gold=e.reference) for j,e in enumerate(examples))
    state=binary_plurality_snapshot(state_id=snapshot['team_state_id'],member_prompts=tuple(snapshot['member_prompts']),
        observations=obs,capabilities=benchmark.capabilities)
    state=replace(state,member_outputs=tuple(tuple(NS(**p) for p in member) for member in snapshot['member_outputs']),
        member_scores=tuple(snapshot['member_scores']))
    diagnosis=BinaryPluralityResponsibilityAnalyzer(benchmark.capabilities).analyze(state,history)
    source=BinaryEvidenceSource(NS(snapshot=lambda:state,examples=examples),history)
    rows={m:source.for_member(state,diagnosis,m) for m in range(5)}
    eligible=tuple(m for m in range(5) if VariableEvidenceFeasibilityV1().feasible(state,diagnosis,m,rows[m]))
    target=TargetPolicyV1().select(state,diagnosis,eligible,history)
    return state,diagnosis,rows,target


def verify_request_firewall(trace,validity,attempts,contract,benchmark,examples,prompt_bank):
    """Check wire identity and benchmark-owned Solver messages without transport."""
    broker=RequestBroker(contract=contract,transport=None,arm='A4',seed=81)
    validity_by_key={r['request_sha256']:r for r in validity}
    attempts_by_no={r['attempt']:r for r in attempts}
    checked=0
    for row in trace:
        lane=None
        if row['role']=='solver':
            v=validity_by_key.get(row['request_sha256'])
            lane=attempts_by_no[row['physical_attempt_no']]['member_realization_lane']
            if v is not None:
                assert lane==v['member_id']
                item=examples[v['split']][v['input_id']].item
                prompt=prompt_bank[v['mutable_prompt_sha256']]
                assert row['request']['messages']==[
                    dict(role='system',content=benchmark.output_contract),
                    dict(role='user',content=benchmark.solver_user_content(prompt,item))]
            else:
                assert any(row['request']['messages']==[
                    dict(role='system',content=benchmark.output_contract),
                    dict(role='user',content=benchmark.solver_user_content(prompt,e.item))]
                    for prompt in prompt_bank.values() for e in examples[row['split']].values())
            checked+=1
        request,key=broker._request_identity(role=row['role'],split=row['split'],
            messages=row['request']['messages'],member_slot=lane)
        assert request==row['request'] and key==row['request_sha256']
    return dict(status='PASS',solver_wire_rows_checked=checked,
        request_identity_rows_checked=len(trace),member_lane_identity_checked=True,
        benchmark_owned_solver_messages_checked=True,optimizer_context_leakage=False,audit_provider_calls=0)


def produce_pilot_report(root,work):
    root=Path(root);work=Path(work);info=read(work/'registration.json');frozen=read(work/'frozen_scope.json')
    run=root/info['active_run'];before=hashes(run);life=read(run/'lifecycle.json')
    assert life['status'] in {'EXECUTION_COMPLETE','EXECUTION_ABORTED'}
    payload=read(root/info['prep']/'prep.json')
    validate_frozen_source(root,payload['source_identity']);verify_source_commit(root,frozen['source_sha'],payload['source_identity'])
    c=read(root/info['binding']);binding=execution_binding(root,c);assert not binding.blockers()
    inventory=read(run/'raw_evidence_inventory.json')
    expected={r['path']:r['sha256'] for r in inventory['files']}
    assert expected=={k:v for k,v in before.items() if k not in {'raw_evidence_inventory.json','SEARCH_COMPLETE_RECEIPT.json'}}
    if life['status']=='EXECUTION_COMPLETE':
        receipt=read(run/'SEARCH_COMPLETE_RECEIPT.json')
        assert receipt['search_closed_forever'] and receipt['raw_inventory_sha256']==before['raw_evidence_inventory.json']
        assert receipt['startup_identity_sha256']==payload['startup_identity_sha256']
        assert receipt['final_team_sha256']==before['final_team_private.json']
        assert receipt['trajectory_sha256']==before['trajectory_private.jsonl']
    trace=lines(run/'provider_trace_private.jsonl');ledger=lines(run/'ledger.jsonl');stages=lines(run/'trajectory_private.jsonl')
    memory_journal=lines(run/'pilot_observation_private.jsonl')
    responses=[r for r in trace if 'response' in r]
    successes=[r for r in ledger if r['kind']=='SUCCESS'];attempts=[r for r in ledger if r['kind']=='ATTEMPT']
    failures=[r for r in ledger if r['kind']=='FAILURE'];cache=[r for r in ledger if r['kind']=='CACHE_HIT']
    sig=lambda r:(r['role'],r['request_sha256'],r.get('response',r)['input_tokens'],r.get('response',r)['output_tokens'])
    assert Counter(map(sig,responses))==Counter(map(sig,successes))
    assert len(attempts)==len(successes)+len(failures)
    assert len(successes)<=c['provider_bounds']['successful_provider_calls'] and len(attempts)<=c['provider_bounds']['transport_attempts']
    assert {r['split'] for r in attempts}<={'optimize','shadow'}
    assert {r['role'] for r in attempts}<={'solver','reflection','pattern_gradient','pattern_cluster'}
    assert all(r['role']=='solver' for r in cache)
    for row in trace:
        fields=generation_request_fields(c,row['role'])
        assert all(row['request'].get(k)==v for k,v in fields.items())
        assert set(row['request'])=={'model','messages',*fields}
    start=read(run/'accounting_start.json');end=read(run/'accounting_end.json');charge=end['charged_total']-start['charged_total']
    assert end['reserved_inflight']==0 and end['by_attempt'][info['attempt']]['charged_total']==charge
    ledger_before=hashes(root/c['token_ledger_directory'])
    with TokenLedger(root/c['token_ledger_directory'],task_sha256=c['task_authorization_sha256'],policy=POLICY_40M) as durable:
        assert durable.view()==end
        reserved={e['reservation_id']:e for e in durable.events if e['kind']=='RESERVE' and e.get('attempt_id')==info['attempt']}
        charges=[dict(**e,role=reserved[e['reservation_id']]['role'],phase=reserved[e['reservation_id']]['stage'])
            for e in durable.events if e['kind']=='CHARGE' and e['reservation_id'] in reserved]
        assert len(charges)==len(attempts) and sum(e['charged'] for e in charges)==charge
        for event,attempt in zip(charges,attempts,strict=True):
            assert event['role']==attempt['role']
            event.update(stage=attempt['stage'],split=attempt['split'],physical_attempt_no=attempt['attempt'])
    assert hashes(root/c['token_ledger_directory'])==ledger_before
    if not failures:assert charge==sum(r['input_tokens']+r['output_tokens'] for r in successes)
    predictions,validity=reconstruct_predictions(trace,ledger)
    logical={(r['mutable_prompt_sha256'],r['input_id'],r['split'],r['member_id']):asdict(predictions[r['request_sha256']]) for r in validity}
    examples=binding.examples('optimize');benchmark=binding.benchmark()
    aliases={e.item.input_id:'o'+str(i) for i,e in enumerate(examples,1)}
    alias=lambda x:aliases[x]
    initial=read(run/'initial_state_private.json') if (run/'initial_state_private.json').exists() else None
    initial_metrics=snapshot_measure(initial,benchmark,examples) if initial else None
    ops=[r for r in stages if r['stage']=='OPPORTUNITY'];evs={r['opportunity_id']:r for r in stages if r['stage']=='EVALUATION'}
    transitions={r['opportunity_id']:r for r in stages if r['stage']=='TRANSITION'}
    summary=read(run/'execution_summary.json') if (run/'execution_summary.json').exists() else None
    final=read(run/'final_state_private.json') if (run/'final_state_private.json').exists() else (list(transitions.values())[-1]['child'] if transitions else initial)
    final_metrics=snapshot_measure(final,benchmark,examples) if final else None
    gradients=[r for r in responses if r['role']=='pattern_gradient'];clusters=[r for r in responses if r['role']=='pattern_cluster']
    reflections=[r for r in responses if r['role']=='reflection']
    gi=ci=0;history=HistoryState();op_start_physical=[]
    who=[];gradient_public=[];gradient_private=[];pattern_public=[];pattern_private=[]
    candidates=[];candidate_private=[];op_rows=[];team_states=[];stop_rows=[];transition_rows=[];cost_rows=[]
    funnel=Counter();prompt_lineage=[];target_sequence=[]
    prompt_bank={text_hash(p):p for p in initial['member_prompts']} if initial else {}
    for row in reflections:
        try:
            p=json.loads(row['response']['text'])
            if isinstance(p,dict) and isinstance(p.get('decision_procedure'),str):
                prompt_bank[text_hash(p['decision_procedure'])]=p['decision_procedure']
            packet=json.loads(row['request']['messages'][0]['content'].rsplit('\n',1)[1])
            prompt_bank[text_hash(packet['current_parent'])]=packet['current_parent']
        except (ValueError,TypeError,KeyError):pass
    private_examples={'optimize':{e.item.input_id:e for e in examples},'shadow':{}}
    if any(r['split']=='shadow' for r in attempts):
        private_examples['shadow']={e.item.input_id:e for e in binding.examples('shadow')}
    firewall_audit=verify_request_firewall(trace,validity,attempts,c,benchmark,private_examples,prompt_bank)
    # This producer audits the observed pre-cluster abort only. Other terminal
    # shapes require their own audited producer; never claim unverified gates.
    assert life['status']=='EXECUTION_ABORTED'
    assert not ops and not evs and not transitions and summary is None
    assert not clusters and not reflections
    # Remaining calls are an explicitly incomplete opportunity, never silently dropped.
    partial_parent=final
    partial_target=None
    if gi<len(gradients) and partial_parent:
        _,diagnosis,source_rows,decision=who_reconstruction(partial_parent,benchmark,examples,history)
        partial_target=decision.selected_member
        who.append(dict(opportunity_index=len(ops)+1,incomplete=True,target_member=partial_target,
            members=[dict(member_id=m,D=s.direct_count,N=s.near_margin_count,C=s.coverage_count,F=s.raw_value,
                current_competence=partial_parent['member_scores'][m],initial_floor=initial['member_scores'][m],
                failure_count=history.failure_counts.get(m,0),discounted_score=decision.target_scores.get(m),
                feasible=m in decision.eligible_members,primary_lane=s.primary_lane) for m,s in diagnosis.responsibility.items()]))
        rows={r.example_id:r for r in source_rows[partial_target]}
        first=gradients[gi]['physical_attempt_no'];op_start_physical.append(first-attempts[first-1]['transport_retry_no'])
        for row in gradients[gi:]:
            p=json.loads(row['request']['messages'][1]['content']);xid=p['example']['example_id']
            assert p['current_member_procedure']==partial_parent['member_prompts'][partial_target]
            value=None
            try:
                value=json.loads(row['response']['text']);assert set(value)=={'gradient'}
                text=value['gradient'];validate_gradient(text,(rows[xid],));status='PASS';category=None
            except (ValueError,TypeError,AssertionError,SearchContractError) as error:
                text=value.get('gradient') if isinstance(value,dict) else None
                status='FAIL';category=str(error) if isinstance(error,SearchContractError) else type(error).__name__
            pub=dict(opportunity_index=len(ops)+1,incomplete=True,example_alias=alias(xid),
                responsibility_labels=p['example']['responsibility_labels'],prediction_valid=p['example']['valid'],
                gradient_sha256=text_hash(text) if isinstance(text,str) else None,length=len(text) if isinstance(text,str) else None,
                contract_status=status,guard_status=status,category=category,provider_request_sha256=row['request_sha256'],
                input_tokens=row['response']['input_tokens'],output_tokens=row['response']['output_tokens'])
            gradient_public.append(pub);gradient_private.append(dict(**pub,exact_generated_gradient=text))
        expected_wrong=[r.example_id for r in source_rows[partial_target] if not r.signals['target_member_correct']]
        actual_ids=[json.loads(r['request']['messages'][1]['content'])['example']['example_id'] for r in gradients[gi:]]
        assert actual_ids==expected_wrong[:len(actual_ids)]
        partial_wrong_count=len(expected_wrong)
        who[-1].update(expected_gradient_count=partial_wrong_count,generated_gradient_count=len(actual_ids),
            ungenerated_gradient_count=partial_wrong_count-len(actual_ids),failure_counts_not_updated=True)
    for i,start_physical in enumerate(op_start_physical,1):
        stop_physical=op_start_physical[i] if i<len(op_start_physical) else len(attempts)+1
        selected_charges=[e for e in charges if start_physical<=e['physical_attempt_no']<stop_physical]
        cost_rows.append(dict(opportunity_index=i,incomplete=i>len(transitions),
            physical_attempts=len(selected_charges),charged_tokens=sum(e['charged'] for e in selected_charges),
            successful_calls=sum(start_physical<=r['physical_attempt_no']<stop_physical for r in responses),
            roles={role:dict(physical_attempts=sum(e['role']==role for e in selected_charges),
                charged_tokens=sum(e['charged'] for e in selected_charges if e['role']==role))
                for role in ('solver','pattern_gradient','pattern_cluster','reflection')}))
    assert hashes(run)==before
    return locals()


def emit_packet(a):
    root,work,info,run=(a[k] for k in ('root','work','info','run'))
    life=a['life'];frozen=a['frozen'];c=a['c'];success=a['successes'];ledger=a['ledger'];memory=a['memory_journal']
    assert life['status']=='EXECUTION_ABORTED' and not a['ops'] and not a['clusters'] and not a['reflections']
    validate_frozen_source(root,a['payload']['source_identity'])
    frozen_prompts=[m['prompt'] for m in read(root/c['initial_team_path'])['members']]
    assert a['initial']['member_prompts']==frozen_prompts==a['final']['member_prompts']
    for m,prompt in enumerate(a['initial']['member_prompts']):
        for j,example in enumerate(a['examples']):
            assert digest(a['initial']['diagnostics']['raw_profiles'][m][j])==digest(a['logical'][(text_hash(prompt),example.item.input_id,'optimize',m)])
    floor=read(run/'initial_competence_floor.json')
    assert floor['member_scores']==a['initial_metrics']['member_correct_counts']
    assert floor['state_id']==a['initial']['team_state_id'] and floor['binding']==c['initial_competence_binding']
    gradient_prompt=read(root/c['gradient_prompt_path'])['prompt']
    for row in a['gradients']:
        packet=json.loads(row['request']['messages'][1]['content'])
        source=a['rows'][packet['example']['example_id']]
        assert digest(packet)==digest(dict(schema=GRADIENT_POLICY['schema'],
            current_member_procedure=a['initial']['member_prompts'][a['partial_target']],example=single_failure_example(source)))
        assert row['request']['messages'][0]['content']==gradient_prompt
    assert len({r['request_sha256'] for r in a['gradients']})==len(a['gradients'])
    assert len(memory)==1 and memory[0]['stage']=='INITIAL_EMPTY_MEMORY'
    assert not memory[0]['memory_state']['private'] and not memory[0]['memory_state']['failures'] and not memory[0]['memory_state']['shared']
    complete=life['status']=='EXECUTION_COMPLETE'
    stop_reason=a['summary']['result']['stop_reason'] if a['summary'] else life.get('stop_category','EXECUTION_ABORTED')
    changed=bool(a['initial'] and a['final'] and a['initial']['member_prompts']!=a['final']['member_prompts'])
    validation_status='DEFERRED_BY_USER_SCOPE' if changed else 'SKIPPED_NO_TEAM_CHANGE'
    contract_failure=not complete and (str(stop_reason).startswith(('PATTERN_','STOP_PATTERN_')))
    classification='VALID_A4_SEED81_PILOT_SEARCH' if complete else ('NOT_EVALUABLE_GENERATED_PATTERN_CONTRACT_FAILURE' if contract_failure else 'HOLD_OPERATIONAL_AUDIT_REQUIRED')
    for name in ('initial','final'):
        snapshot=a[name];metrics=a[name+'_metrics']
        if snapshot is None:continue
        metrics.pop('mean_disagreement',None)
        n=metrics['count'];metrics['member_accuracies']=[x/n for x in metrics['member_correct_counts']]
        metrics['invalid_rates_by_member']=[x/n for x in metrics['invalid_count_by_member']]
        class_counts=[];pair_fractions=[]
        for state in snapshot['diagnostics']['team_states']:
            counts=Counter(answer for answer,valid in zip(state['team_answers'],state['team_validity'],strict=True) if valid)
            class_counts.append(len(counts));valid=sum(counts.values())
            if valid>=2:pair_fractions.append(1-sum(x*(x-1) for x in counts.values())/(valid*(valid-1)))
        metrics['mean_distinct_valid_equivalence_classes']=sum(class_counts)/n
        metrics['mean_valid_pair_disagreement']=sum(pair_fractions)/len(pair_fractions) if pair_fractions else None
        metrics['disagreement_definition']='mean unequal-equivalence fraction over valid member pairs; rows with fewer than two valid outputs excluded'
    if a['partial_target'] is not None and not a['op_rows']:
        selected=next(m for m in a['who'][-1]['members'] if m['member_id']==a['partial_target'])
        a['op_rows'].append(dict(opportunity_index=1,incomplete=True,target_member=a['partial_target'],
            raw_F=selected['F'],discounted_score=selected['discounted_score'],wrong_count=a.get('partial_wrong_count'),
            gradient_count=len(a['gradients']),gradient_passes=sum(r['contract_status']=='PASS' for r in a['gradient_public']),
            gradient_failures=sum(r['contract_status']=='FAIL' for r in a['gradient_public']),
            pattern_count=0,selected_pattern_F='NOT_REACHED',selected_support_count='NOT_REACHED',
            memory_reads=0,memory_writes=0,proposals=0,valid=0,local_positive=0,local_neutral=0,local_negative=0,
            exported=0,TeamProbe_pass=0,Full_pass=0,Shadow_pass=0,commit=0,
            team_vote_before=a['initial_metrics']['vote_correct_count'],team_vote_after=a['final_metrics']['vote_correct_count']))
    for key in ('optimizer_generations','changed_valid_scored','exported','teamprobe','teamprobe_pass',
        'full','full_pass','shadow','shadow_pass','commit','patterns','local_positive','local_neutral','local_negative'):
        a['funnel'].setdefault(key,0)
    a['funnel']['gradient_calls']=len(a['gradients']);a['funnel']['cluster_calls']=len(a['clusters'])
    # The raw tree stays immutable. Human analysis is a separately sealed bundle.
    bundle=work/'analysis_bundle';assert not bundle.exists();bundle.mkdir()
    write(bundle,'gradients.json',a['gradient_private'])
    write(bundle,'patterns.json',a['pattern_private'])
    write(bundle,'candidates.json',a['candidate_private'])
    write(bundle,'memory_state_trajectory.json',memory)
    write(bundle,'prompt_lineage.json',a['prompt_lineage'])
    write(bundle,'initial_final_prompts.json',dict(initial=a['initial']['member_prompts'] if a['initial'] else None,
        final=a['final']['member_prompts'] if a['final'] else None))
    (bundle/'README.md').write_bytes(b'# Local immutable Pilot analysis bundle\n\nExact generated gradients, generalized gradients, candidate procedures/diffs and actual bounded Memory reads/states. No raw Shadow, Validation or Test samples. Raw Optimize provider evidence stays in the runtime journal. This descriptive packet does not establish component efficacy or generalization.\n')
    write(bundle,'sha256_manifest.json',build_sha256_manifest(bundle))
    bundle_ref=dict(path=bundle.relative_to(root).as_posix(),sha256_manifest_sha256=sha256((bundle/'sha256_manifest.json').read_bytes()).hexdigest())
    report=root/info['execution_report'];assert not report.exists();report.mkdir()
    common=dict(attempt_id=info['attempt'],source_sha=frozen['source_sha'],arm='A4',seed=81,
        Optimize_count=60,Shadow_count=40,scientific_efficacy='NOT_ESTIMATED',validation_model_calls=0,test_model_calls=0,
        raw_diagnostic_calls=0,llm_judge_calls=0,pushed=False)
    usage={role:dict(successful_calls=sum(r['role']==role for r in success),
        physical_attempts=sum(r['role']==role for r in a['attempts']),transport_failures=sum(r['role']==role for r in a['failures']),
        cache_hits=sum(r['role']==role for r in a['cache']),
        charged_input_tokens=sum(r['input_tokens'] for r in a['charges'] if r['role']==role),
        charged_output_tokens=sum(r['output_tokens'] for r in a['charges'] if r['role']==role))
        for role in ('solver','pattern_gradient','pattern_cluster','reflection')}
    memory_events=[];prior=None
    for index,row in enumerate(memory):
        state=row['memory_state'];audit=state['audit']
        public=dict(event_index=index,stage=row['stage'],context=row['context'],memory_state_sha256=digest(state),
            audit=audit,revision=state['revision'],sequence=state['sequence'],risk_clock=state['clock'],
            storage_utilization=dict(private_success=len(state['private']),private_failure=len(state['failures']),shared_risk=len(state['shared'])))
        if prior:
            public['counter_deltas']={k:audit[k]-prior['audit'].get(k,0) for k in (
                'success_reads','failure_reads','shared_reads','success_writes','failure_writes','shared_writes',
                'failure_memory_evictions','failure_to_shared_promotions','shared_risk_created','shared_risk_updated','shared_risk_evicted')}
            for name in ('private','failures','shared'):
                old={e['memory_id'] for e in prior[name]};new={e['memory_id'] for e in state[name]}
                public[name+'_added_ids']=sorted(new-old);public[name+'_removed_ids']=sorted(old-new)
        if row['stage']=='MEMORY_READ':
            data=row['data'];public.update(member=data['member'],lane=data['lane'],context_chars=data['context_chars'],
                visible_sha256=digest(data['visible']),
                read_ids={k:[e['memory_id'] for e in values] for k,values in data['entries'].items()},
                read_entries=sum(len(values) for values in data['entries'].values()))
        memory_events.append(public);prior=state
    final_memory=memory[-1]['memory_state']['audit'] if memory else None
    if a['summary']:assert final_memory==a['summary']['memory_audit']
    member_rows=[]
    for m in range(5):
        targets=[r for r in a['op_rows'] if r['target_member']==m]
        member_rows.append(dict(member_id=m,opportunities=len(targets),
            completed_opportunities=0,times_targeted=sum(r['target_member']==m for r in a['who']),
            opportunity_share=sum(r['target_member']==m for r in a['who'])/len(a['who']) if a['who'] else 0,
            initial_correct=a['initial_metrics']['member_correct_counts'][m] if a['initial_metrics'] else None,
            final_correct=a['final_metrics']['member_correct_counts'][m] if a['final_metrics'] else None,
            proposals=sum(r.get('optimizer_generations',0) for r in targets),
            scored=sum(r.get('changed_valid_scored',0) for r in targets),exported=sum(r.get('exported',0) for r in targets),
            commits=sum(r.get('commit',0) for r in targets),
            private_failure_writes=0,private_success_writes=0,shared_risk_contributions=0,team_gain_events=0,
            private_failure_count=final_memory['private_failure_count_by_member'][m] if final_memory else 0))
    final_status=dict(**common,classification=classification,execution_status=life['status'],stop_reason=stop_reason,
        base_commit=frozen['base_commit'],executable_commit=frozen['source_sha'],
        preexecution_commit=read(work/'ready.json')['repository_commit'],
        opportunities_completed=len(a['transitions']),opportunities_built=len(a['ops']),
        target_sequence=a['target_sequence'],incomplete_target_member=a['partial_target'],
        commits=a['funnel'].get('commit',0),initial_Optimize_metrics=a['initial_metrics'],final_Optimize_metrics=a['final_metrics'],
        team_changed=changed,validation_status=validation_status,provider_calls=len(a['attempts']),provider_usage_by_role=usage,
        charged_tokens=a['charge'],cumulative_charged=a['end']['charged_total'],remaining_tokens=a['end']['remaining'],
        candidate_funnel=dict(a['funnel']),memory_final_audit=final_memory,analysis_bundle=bundle_ref,
        full_current_suite=frozen['full_current_counts'],historical_private_tests='NOT_RUN',
        other_arms_executed=False,other_seeds_executed=False,scientific_reruns=0,automatic_operational_retries_used=0,
        future_execution_requires_new_exact_scope=True,source_closure_matches=True,
        accounting_integrity='PASS',pattern_or_gradient_contract_failure=contract_failure)
    final_status.update(pilot_search_complete=complete,
        final_team_disposition='FINAL_TEAM_FROZEN' if complete else 'NO_PILOT_FINAL_TEAM_INITIAL_OR_COMMITTED_PREFIX_RETAINED',
        final_metrics_scope='FINAL_PILOT_OPTIMIZE' if complete else 'DEPLOYED_PREFIX_ONLY_INCOMPLETE_PILOT',
        expected_incomplete_gradient_count=a.get('partial_wrong_count'),
        task_stop_reason='STOP_FOR_USER_TRAINING_PROCESS_ANALYSIS' if complete else 'USER_SCIENTIFIC_POLICY_DECISION_REQUIRED',
        Optimize_vote_delta=a['final_metrics']['VoteAcc']-a['initial_metrics']['VoteAcc'] if a['initial_metrics'] else None,
        Optimize_oracle_delta=a['final_metrics']['OracleAcc']-a['initial_metrics']['OracleAcc'] if a['initial_metrics'] else None,
        all_future_execution_scopes_closed=True)
    write(report,'final_status.json',final_status)
    write(report,'execution_identity.json',dict(**common,startup_identity_sha256=frozen['startup_identity_sha256'],
        base_commit=frozen['base_commit'],preexecution_commit=final_status['preexecution_commit'],
        source_identity_hashes={k:v for k,v in a['payload']['source_identity'].items() if k.endswith('_hash')},
        binding_sha256=frozen['binding_sha256'],frozen_initial_state_empty=True,cache_namespace=c['cache_namespace'],
        execution_entrypoint='scripts/run_experiment.py',composition='build_current_team_prompt_search',scientific_diff='EMPTY'))
    write(report,'provider_usage.json',dict(**common,roles=usage,physical_attempts=len(a['attempts']),
        successful_calls=len(success),transport_failures=len(a['failures']),logical_solver_evaluations=len(a['validity']),
        logical_solver_requests=len(a['predictions']),
        raw_invalid_solver_attempts=sum(r['kind']=='SEMANTIC_ATTEMPT' and not r['prediction_valid'] for r in ledger),
        recovered_logical_solver_requests=sum(r['recovered_invalid'] for r in a['validity']),
        terminal_invalid_solver_evaluations=sum(r['terminal_invalid'] for r in a['validity']),
        charged_tokens=a['charge'],charge_measurement='RESERVATION_V2_OPERATIONAL_ACCOUNTING_NOT_PROVIDER_BILLING_PROOF'))
    write(report,'accounting_integrity_audit.json',dict(**common,status='PASS',classified_physical_attempts=len(a['charges']),
        unclassified_charged_calls=0,reserved_inflight=0,charge_sum=sum(r['charged'] for r in a['charges']),
        attempt_charge=a['charge'],fallback_charged=sum(r['charged'] for r in a['charges'] if not r['reliable_usage']),
        cumulative_charged=a['end']['charged_total'],remaining=a['end']['remaining'],cumulative_authorized=40_000_000,
        hash_chain_replay='PASS',provider_response_success_ledger_match='PASS',frozen_wire_policy='PASS',
        source_closure='PASS',terminal_inventory='PASS',raw_tree_unchanged=True))
    write(report,'solver_firewall_audit.json',dict(**common,**a['firewall_audit']))
    diagnostics=[r for r in ledger if r['kind']=='OPTIMIZER_GENERATION_DIAGNOSTICS']
    assert len(diagnostics)==sum(r['role']!='solver' for r in success)
    assert all(r['enable_thinking'] is False and r['nonthinking_wire_confirmed'] is True for r in diagnostics)
    write(report,'nonthinking_wire_audit.json',dict(**common,status='PASS',
        optimizer_evidence_levels=dict(Counter(r['nonthinking_evidence_level'] for r in diagnostics)),
        optimizer_wire_requests_checked=len(diagnostics),reasoning_tokens_not_inferred_as_zero=True,
        raw_response_metadata_counts=dict(reported_reasoning_tokens=sum(r.get('reasoning_tokens') is not None for r in diagnostics),
            reported_reasoning_content_chars=sum(r.get('reasoning_content_chars') is not None for r in diagnostics))))
    closure=read(work/'authorization_closure.json')
    assert closure['closed'] and closure['single_use_consumed'] and not closure['pilot_authorized']
    write(report,'authorization_closure.json',closure)
    forensic=read(work/'gradient_forensic.json')
    assert len(forensic)==len(a['gradients'])
    assert len(forensic)==4 and all(r['contract_status']=='PASS' for r in forensic[:3])
    assert forensic[-1]['length']==267 and forensic[-1]['has_numeric_content'] is True
    assert forensic[-1]['gradient_sha256']=='666f17d12d303afe5e5e6685cfbc0afc9d345f9756621c5ffed28a61dd6e4c01'
    assert [r['gradient_sha256'] for r in forensic]==[r['gradient_sha256'] for r in a['gradient_public']]
    write(report,'gradient_contract_forensic.json',dict(**common,rows=forensic,
        conclusion='FROZEN_DIGIT_PROHIBITION_ENFORCED',
        root_category='GENERATED_CONTRACT_NONCOMPLIANCE',proven_implementation_invalid=False,
        automatic_retry_eligible=False,numeric_literal_context='GENERAL_MATHEMATICAL_RULE',
        problem_specific_numeric_copying_established=False,
        required_user_decision='ANY_PROMPT_OR_GUARD_AMENDMENT_REQUIRES_FRESH_IDENTITY_AND_EXACT_AUTHORIZATION',
        guard_or_prompt_changed=False,gradient_hard_length_limit=400))
    write(report,'team_state_trajectory.json',dict(**common,initial=a['initial_metrics'],final=a['final_metrics'],trajectory=a['team_states'],
        initial_state_id=a['initial']['team_state_id'],final_deployed_prefix_state_id=a['final']['team_state_id'],
        initial_prompt_hashes=[text_hash(p) for p in frozen_prompts],committed_states=0,
        final_metrics_scope='DEPLOYED_PREFIX_ONLY_INCOMPLETE_PILOT'))
    write(report,'initial_team_audit.json',dict(**common,status='PASS',initial_state_id=a['initial']['team_state_id'],
        initial_prompt_hashes=[text_hash(p) for p in a['initial']['member_prompts']],
        initial_team_sha256=digest(a['initial']['member_prompts']),
        final_deployed_prefix_team_sha256=digest(a['final']['member_prompts']),
        persisted_profiles_match_resolved_provider_predictions=True,initial_floor_matches=True,
        logical_evaluations=300,independent_metric_reconstruction=True,
        initial_competence_support_identity=floor['binding']['support_identity']))
    write(report,'responsibility_trajectory.json',dict(**common,trajectory=a['who'],reconstruction='PASS',
        pattern_and_memory_did_not_determine_target=True))
    write(report,'gradient_trajectory.json',dict(**common,rows=a['gradient_public'],
        contract_passes=sum(r['contract_status']=='PASS' for r in a['gradient_public']),
        contract_failures=sum(r['contract_status']=='FAIL' for r in a['gradient_public']),
        exact_texts=bundle_ref,semantic_quality='USER_REVIEW_PENDING_NO_LLM_JUDGE'))
    write(report,'pattern_trajectory.json',dict(**common,rows=a['pattern_public'],
        unintegrated_cluster_calls=len(a['clusters'])-a['ci'],partial_opportunity=not complete,exact_texts=bundle_ref))
    write(report,'memory_trajectory.json',dict(**common,events=memory_events,final_audit=final_memory,
        actual_retrieval_invocations=sum(r['stage']=='MEMORY_READ' for r in memory),
        per_generation_and_opportunity_restoration_available=bool(a['reflections'] or a['ops']),
        generation_and_opportunity_states='NOT_REACHED' if not a['reflections'] and not a['ops'] else 'JOURNALED',
        full_states=bundle_ref,memory_llm_calls=0))
    (report/'candidate_trajectory.jsonl').write_bytes(''.join(json.dumps(r,sort_keys=True)+'\n' for r in a['candidates']).encode())
    write(report,'candidate_funnel.json',dict(**common,counts=dict(a['funnel']),
        local_positive_did_not_gate_pool=True,all_changed_valid_unique_scored_eligible=True,
        candidate_pool_observed=False,admission_contract_status='FROZEN_CONTRACT_NOT_EXERCISED',
        stage_absence='NOT_REACHED' if not a['candidates'] else 'See per-candidate statuses',
        contract_failed_incomplete_opportunity=contract_failure))
    write(report,'stage_disposition.json',dict(**common,
        initial_Optimize='COMPLETE' if a['initial'] else 'INCOMPLETE',
        gradient_extraction='CONTRACT_FAILURE_INCOMPLETE' if contract_failure else 'See gradient_trajectory.json',
        Pattern='NOT_REACHED' if not a['clusters'] else 'See pattern_trajectory.json',
        Memory_reads='NOT_REACHED' if not any(r['stage']=='MEMORY_READ' for r in memory) else 'See memory_trajectory.json',
        Layer1='NOT_REACHED' if not a['reflections'] else 'See candidate_trajectory.jsonl',
        TeamProbe='NOT_REACHED' if not a['funnel'].get('teamprobe') else 'See candidate_trajectory.jsonl',
        Full='NOT_REACHED' if not a['funnel'].get('full') else 'See candidate_trajectory.jsonl',
        Shadow='NOT_REACHED' if not a['funnel'].get('shadow') else 'AGGREGATE_ONLY',
        commit='NOT_REACHED' if not a['transition_rows'] else 'See transition_trace.json'))
    write(report,'transition_trace.json',dict(**common,rows=a['transition_rows'],
        initial_competence_floor_immutable=True,atomic_parent_child_identity_checked=True))
    write(report,'stop_trace.json',dict(**common,rows=a['stop_rows'],stop_reason=stop_reason,
        scientific_stop_reached=complete,scientific_stopper='team_epoch_no_commit_v1',patience=2,
        operational_bound_was_not_scientific_stop=True,no_efficacy_adaptation=True))
    write(report,'prompt_lineage.json',dict(**common,rows=a['prompt_lineage'],exact_prompts_and_diffs=bundle_ref))
    role_stage=defaultdict(lambda:dict(physical_attempts=0,charged_input_tokens=0,charged_output_tokens=0))
    for row in a['charges']:
        key=row['stage']+':'+row['role'];role_stage[key]['physical_attempts']+=1
        role_stage[key]['charged_input_tokens']+=row['input_tokens'];role_stage[key]['charged_output_tokens']+=row['output_tokens']
    first=a['op_start_physical'][0]
    initial_charges=[r for r in a['charges'] if r['physical_attempt_no']<first]
    for index,row in enumerate(a['cost_rows']):
        lo=a['op_start_physical'][index];hi=a['op_start_physical'][index+1] if index+1<len(a['op_start_physical']) else len(a['attempts'])+1
        receipts=[r for r in a['responses'] if lo<=r['physical_attempt_no']<hi]
        row.update(input_tokens=sum(r['response']['input_tokens'] for r in receipts),
            output_tokens=sum(r['response']['output_tokens'] for r in receipts),cache_hits=0,
            cumulative_charged=a['start']['charged_total']+sum(r['charged'] for r in a['charges'] if r['physical_attempt_no']<hi))
    write(report,'cost_trajectory.json',dict(**common,role_stage=dict(role_stage),opportunities=a['cost_rows'],
        initialization=dict(physical_attempts=len(initial_charges),charged_tokens=sum(r['charged'] for r in initial_charges),
            input_tokens=sum(r['input_tokens'] for r in initial_charges),output_tokens=sum(r['output_tokens'] for r in initial_charges)),
        charge_events=[dict(index=i,role=r['role'],stage=r['stage'],charged_tokens=r['charged'],usage_reliable=r['reliable_usage']) for i,r in enumerate(a['charges'])],
        efficiency_per_commit=a['charge']/a['funnel']['commit'] if a['funnel'].get('commit') else 'NOT_ESTIMATED',
        candidate_per_proposal='NOT_ESTIMATED',export_per_proposal='NOT_ESTIMATED',
        TeamProbe_pass_per_export='NOT_ESTIMATED',Full_pass_per_attempt='NOT_ESTIMATED',Shadow_pass_per_attempt='NOT_ESTIMATED',
        commit_per_completed_opportunity='NOT_ESTIMATED',tokens_per_completed_opportunity='NOT_ESTIMATED',
        efficiency_interpretation='DESCRIPTIVE_COST_ONLY_NO_COMPARATIVE_EFFICACY'))
    write(report,'module_behavior_summary.json',dict(**common,WHO='RECONSTRUCTED_FROM_FIXED_PARENT',
        gradients=dict(calls=len(a['gradients']),contracts_passed=sum(r['contract_status']=='PASS' for r in a['gradient_public']),
            contracts_failed=sum(r['contract_status']=='FAIL' for r in a['gradient_public'])),
        Pattern=dict(cluster_calls=len(a['clusters']),integrated_opportunities=len(a['pattern_public']),
            pattern_count=sum(r['pattern_count'] for r in a['pattern_public']),
            shared_patterns=sum(sum(p['support_count']>1 for p in r['patterns']) for r in a['pattern_public'])),
        Memory=final_memory,Layer1=dict(proposals=a['funnel'].get('optimizer_generations',0)),
        team_gates=dict(a['funnel']),meaning='OBSERVED_PROCESS_ONLY'))
    write(report,'governance_audit.json',dict(**common,status='PASS',validation_status=validation_status,
        Validation_raw_sample_access=0,Test_raw_sample_access=0,roles_and_splits_frozen=True,
        Optimize_Shadow_memberships_unchanged=True,source_startup_identity_frozen=True,
        initial_team_frozen=True,initial_Memory_empty=True,no_cross_attempt_realization_reuse=True,
        historical_private_tests='NOT_RUN',full_historical_replay_pass=False,
        publication_content='HASHES_COUNTERS_CATEGORIES_AGGREGATE_METRICS',analysis_bundle=bundle_ref))
    for name,rows in (('opportunity_summary.csv',a['op_rows']),('member_training_summary.csv',member_rows)):
        fields=list(dict.fromkeys(k for row in rows for k in row)) if rows else ['opportunity_index','status']
        with (report/name).open('x',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(rows)
    readme=f'''# A4 Seed81 Gradient Pattern Memory Pilot search audit

Status: **{classification}**. Stop: `{stop_reason}`. This is a descriptive training-process run; component efficacy and generalization are **NOT_ESTIMATED**.

Optimize60 and private Shadow40 use existing frozen memberships. Completed opportunities: {len(a['transitions'])}; built: {len(a['ops'])}; target sequence: {a['target_sequence']}; commits: {a['funnel'].get('commit',0)}. Incomplete target: {a['partial_target']}.

Initial Optimize metrics: `{json.dumps(a['initial_metrics'],sort_keys=True)}`.

Final deployed-prefix Optimize metrics: `{json.dumps(a['final_metrics'],sort_keys=True)}`.

Provider calls: {len(a['attempts'])}; charged tokens: {a['charge']}; cumulative: {a['end']['charged_total']}; remaining: {a['end']['remaining']} of the durable 40M authorization. Roles and detailed stages are in [provider usage](provider_usage.json) and [cost trajectory](cost_trajectory.json).

Validation: `{validation_status}` (0 calls). Test: sealed (0 calls). Other arms/seeds, raw diagnostic and LLM judge: 0. No tuning or scientific rerun. No push.

Executable source: `{frozen['source_sha']}`. Preexecution commit: `{final_status['preexecution_commit']}`. Base: `{frozen['base_commit']}`. Full current suite: {frozen['full_current_counts']['passed']} passed, {frozen['full_current_counts']['skipped']} skipped. Historical private tests: NOT_RUN; full historical replay PASS is not claimed.

The [analysis index](analysis_index.md) links the module traces and explicit [stage dispositions](stage_disposition.json). Exact gradients, procedures/diffs and Memory read sets/states are in the ignored local bundle `{bundle_ref['path']}`; its SHA manifest hash is `{bundle_ref['sha256_manifest_sha256']}`. Public evidence contains hashes, counters, categories and aggregate metrics.

This attempt stopped before a complete production opportunity. Three gradients passed; the fourth had 267 characters and violated the frozen digit prohibition. Its numeric literal occurred in a general mathematical rule; problem-specific numeric copying is not established. The guard enforced its frozen implementation. This is not proven implementation invalidity and does not qualify for an automatic operational retry. The 400-character limit, prompt, guard, method, models and stopping are unchanged. [Forensic evidence](gradient_contract_forensic.json) retains hashes and categories.

Search is closed with incomplete Pilot status. No scientifically completed final team is claimed; final metrics above describe the unchanged deployed prefix. [All one-shot scopes are closed](authorization_closure.json), including the unused conditional operational retry. A prompt or guard amendment requires user decision, a fresh identity/freeze and exact authorization under the user task sections 26, 30 and 106. Continuing training requires a new scope. No API calls were made by the report audit.
'''
    (report/'README.md').write_bytes(readme.encode())
    (report/'analysis_index.md').write_bytes(('''# Training-process analysis index

1. [Execution and stopping](final_status.json), [stop reconstruction](stop_trace.json), [team states](team_state_trajectory.json).
2. [WHO: all five members](responsibility_trajectory.json), [opportunities](opportunity_summary.csv), [member summary](member_training_summary.csv).
3. [Per-example gradients](gradient_trajectory.json), [all Patterns / same-F / representatives / preservation / transition](pattern_trajectory.json).
4. [Memory actual reads, writes, updates and evictions](memory_trajectory.json). Exact states are restorable from the local bundle after every generation and opportunity.
5. [All candidate generations](candidate_trajectory.jsonl), [pool and gate funnel](candidate_funnel.json), [prompt lineage](prompt_lineage.json), [atomic transitions](transition_trace.json).
6. [Provider usage](provider_usage.json), [charge timeline](cost_trajectory.json), [ledger audit](accounting_integrity_audit.json), [scope audit](governance_audit.json).

An absent stage means NOT_REACHED. Generated contract failure is retained as incomplete execution and does not authorize regeneration, loosening guards or an efficacy-driven rerun. Exact-text bundle location and integrity hash are recorded in final_status.json. No Shadow/Validation/Test sample data are copied into that bundle.
''').encode())
    assert not scan_sanitized_artifacts(report)
    assert hashes(run)==a['before']
    write(report,'sha256_manifest.json',build_sha256_manifest(report))
    write(work,'owner_audit_summary.json',dict(status='PASS',classification=classification,
        physical_calls=len(a['attempts']),charged_tokens=a['charge'],opportunities_completed=len(a['transitions']),
        target_sequence=a['target_sequence'],contract_failure=contract_failure,analysis_bundle=bundle_ref,
        report_path=report.relative_to(root).as_posix(),runtime_tree_unchanged=True))
    return final_status
