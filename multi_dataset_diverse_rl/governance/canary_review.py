"""Read-only execution review barriers; no search, provider or selection authority."""
from collections import Counter
import json
import time

from ..persistence.durable_io import atomic_write_json, read_json
from ..search.system_prompt import solver_messages
from ..benchmarks.math_solver_decoding import generation_request_fields
from .startup_identity import canonical_sha256
from ..search.schemas import SearchContractError

STRUCTURED_POLICY=dict(identity='RESPONSIBILITY_REPAIR_CANARY_REVIEW_V4',
    stages=['INITIAL_SOLVER_PROFILE','FIRST_COMPLETE_OPPORTUNITY'],
    owner_review_required=True,maximum_wait_seconds=3600,
    extra_provider_calls=0,total_max_opportunities=5,
    scientific_state_reset=False,efficacy_based_stopping=False,
    unreached_branch='NOT_OBSERVED_NO_FORCED_GENERATION')

def trace_rows(run_root):
    path=run_root/'provider_trace_private.jsonl'
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]


def initial_audit(contract,composed,run_root):
    audit=profile_audit(contract,composed.state.solver,run_root,
        ((member,composed.state.prompts[member],example,profile)
            for member,profiles in composed.state.profiles.items()
            for example,profile in zip(composed.state.examples,profiles,strict=True)))
    state=composed.state.snapshot()
    if len(composed.memory.competence)!=5:
        raise SearchContractError('CANARY_INITIAL_COMPETENCE_MEMORY_MISSING')
    return {**audit,'member_scores':state.member_scores,'initial_state_id':state.team_state_id,
        'initial_memory_entries':5,'validation_calls':0,'test_calls':0}


def profile_audit(contract,solver,run_root,profiles):
    """Verify every physical realization, including immutable imported prefixes."""
    from ..search.solver_execution import next_capacity, frozen_execution_policy
    from ..benchmarks.math_structured_answer import classify_prediction
    execution=frozen_execution_policy(contract)
    raw=trace_rows(run_root)
    bykey={(r['request_sha256'],r['semantic_attempt_no']):r for r in raw
        if r['role']=='solver' and 'response' in r}
    statuses=Counter();caps=Counter();imported=physical=logical=0
    for member,prompt,example,profile in profiles:
        source=profile['solver_trajectory']['source'];key=source['request_sha256']
        if (source['member_id']!=member or source['example_id']!=example.item.input_id or source['split']!='optimize'):
            raise SearchContractError('CANARY_INITIAL_PROVENANCE_MISMATCH')
        result=solver.broker.durable_cache.get(key)
        if result is None or canonical_sha256(result['resolved_prediction'])!=canonical_sha256(profile['prediction']):
            raise SearchContractError('CANARY_INITIAL_CACHE_PROFILE_MISMATCH')
        previous=None
        for ordinal,realization in enumerate(result['original_realizations'],1):
            cap=next_capacity(execution,previous)
            expected=dict(model=contract['models']['solver'],**generation_request_fields(contract,'solver'),
                messages=solver_messages(prompt,solver.benchmark.format_input(example.item)))
            expected['max_tokens']=cap
            record=bykey[(realization['request_sha256'],ordinal)];physical+=1
            if (record['request']!=expected or record['response']['text']!=realization['text']
                    or record['response']['finish_reason']!=realization['finish_reason']
                    or realization['output_capacity_tokens']!=cap):
                raise SearchContractError('CANARY_INITIAL_DISPATCH_MISMATCH')
            caps[str(cap)]+=1
            previous=classify_prediction(realization['text'],realization['finish_reason'])
        statuses[profile['solver_trajectory']['solution_status']]+=1;logical+=1
    return dict(logical_profiles=logical,new_physical_realizations=physical,
        reused_physical_realizations=imported,output_capacity_counts=dict(caps),
        solution_status_counts=dict(statuses),validation_calls=0,test_calls=0,
        audit='EXACT_STRUCTURED_SYSTEM_RAW_USER_MESSAGES_AND_PRIVATE_RECEIPTS')


def opportunity_audit(contract,composed,run_root,opportunity):
    rows=trace_rows(run_root);gradient=composed.opportunities.patterns.extractor.provider
    membership=opportunity.evaluation_plan['evidence_audit']['memberships']
    groups=list(map(set,membership.values()))
    if any(a&b for i,a in enumerate(groups) for b in groups[:i]):
        raise SearchContractError('CANARY_EVIDENCE_ROLE_OVERLAP')
    examples={e.item.input_id:e for e in composed.state.examples}
    diagnostics=[row for row in rows if row['role']=='pattern_gradient' and 'response' in row]
    wrong={r.example_id for r in opportunity.evaluation_plan['evidence_universe']
        if not r.signals['target_member_correct']}
    observed=[]
    for row in diagnostics:
        packet=json.loads(row['request']['messages'][1]['content']);e=packet['example'];xid=e['example_id']
        if (packet['current_system_prompt']!=opportunity.parent_prompt.to_dict() or packet['target_member']!=opportunity.target_member or xid not in wrong
                or e['reference_solution']['text']!=examples[xid].reference_solution[:4096]
                or e['reference_solution']['source']!='dataset_worked_solution'
                or e['solver_trajectory']['source']['member_id']!=opportunity.target_member):
            raise SearchContractError('CANARY_GRADIENT_EVIDENCE_MISMATCH')
        observed.append(xid)
    if set(observed)!=wrong or any(not 1<=n<=3 for n in Counter(observed).values()):
        raise SearchContractError('CANARY_INDEPENDENT_GRADIENT_MEMBERSHIP_MISMATCH')
    mutation_problems=[r.signals['input_payload'] for r in opportunity.evidence.mutation_evidence]
    mutations=[row for row in rows if row['role']=='reflection']
    for row in mutations:
        text=row['request']['messages'][0]['content']
        packet=json.loads('{'+text.split('\n{',1)[1])
        if [r['problem'] for r in packet['current_panel_observations']]!=mutation_problems:
            raise SearchContractError('CANARY_MUTATION_MEMBERSHIP_MISMATCH')
    statuses=Counter(e.record['status'] for e in composed.memory.private)
    return dict(target_member=opportunity.target_member,membership_disjoint=True,
        gradient_diagnostics=len(wrong),gradient_physical_draws=len(observed),proposal_generations=len(mutations),
        probe_candidates=len(composed.evaluation.provider.probed),
        full_candidates=len(composed.evaluation.provider.fulled),memory_status_counts=dict(statuses),
        branch_observation=dict(gradient='OBSERVED',mutation='OBSERVED' if mutations else 'NOT_OBSERVED',
            full='OBSERVED' if composed.evaluation.provider.fulled else 'NOT_OBSERVED'),
        observed_flow='ACTUAL_PROVIDER_REQUESTS_AND_COMMITTED_RUNTIME_STATE',
        validation_calls=0,test_calls=0,quality_claim='OWNER_REVIEW_NOT_CAUSAL_EFFICACY_PROOF')


def review(stage,audit,*,contract,payload,run_root):
    policy=contract.get('canary_review_policy')
    if policy != STRUCTURED_POLICY or stage not in policy['stages']:
        raise SearchContractError('CANARY_REVIEW_POLICY_NOT_FROZEN')
    receipt=dict(stage=stage,attempt_id=contract['execution_attempt_id'],
        startup_identity_sha256=payload['startup_identity_sha256'],audit=audit,
        status='AWAITING_OWNER_REVIEW')
    receipt['review_identity_sha256']=canonical_sha256(receipt)
    atomic_write_json(run_root/(stage+'.review_pending.json'),receipt)
    decision=run_root/(stage+'.owner_review.json')
    deadline=time.monotonic()+policy['maximum_wait_seconds']
    while not decision.exists():
        if time.monotonic()>=deadline:raise SearchContractError('CANARY_OWNER_REVIEW_TIMEOUT')
        time.sleep(1)
    value=read_json(decision)
    if (value.get('approved') is not True or value.get('scientific_method_changed') is not False
            or value.get('review_identity_sha256')!=receipt['review_identity_sha256']
            or value.get('startup_identity_sha256')!=payload['startup_identity_sha256']):
        raise SearchContractError('CANARY_OWNER_REVIEW_NOT_APPROVED')
    atomic_write_json(run_root/(stage+'.review_pass.json'),{**receipt,'status':'OWNER_REVIEW_PASS',
        'owner_review_sha256':canonical_sha256(value)})
