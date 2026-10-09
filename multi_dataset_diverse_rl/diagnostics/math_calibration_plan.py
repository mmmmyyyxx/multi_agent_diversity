"""Independent calibration design and fake-only execution, no paid runner."""
from collections import Counter,defaultdict
from copy import deepcopy
import math,hashlib

from .math_baseline_calibration import (digest,NATIVE_ID,BOXED_ID,DIAGNOSTIC_ID,
    native_extract,primary_correctness,diagnostic_extract,grade_diagnostic,paired_comparison)
from ..benchmarks.experiment_splits import quotas,subject
from ..search.system_prompt import SystemPrompt,solver_messages
from ..governance.token_accounting import serialized_request

PLAN_ID='MATH_BASELINE_PAIRED_CALIBRATION_PLAN_V1'
SAMPLING=dict(enable_thinking=False,temperature=0.2,top_p=0.8,top_k=20,min_p=0,
    presence_penalty=0,frequency_penalty=0,max_output_tokens=3600)
ROLE='You are a helpful assistant.'
STRATEGY='Answer the question.'
ANSWERS={
    'A':'On the last line, write "Final answer:" followed by your mathematical answer.',
    'B':'End your response with a final answer on a separate line.\n'
        'The line must start with the exact text "Final answer: "\n'
        'followed by only the mathematical result.\n'
        'Do not use a Markdown heading, bullet, or bold formatting\n'
        'for this line. Do not write anything after it.',
    'C':'On the last line, write only your final mathematical answer\n'
        'in the form \\boxed{...}. Do not write anything after it.'}


def request_for(arm,problem,capacity=3600):
    if arm not in ANSWERS or capacity not in (3600,6144):raise ValueError('CALIBRATION_REQUEST_POLICY_MISMATCH')
    prompt=SystemPrompt(ROLE,STRATEGY,ANSWERS[arm])
    return dict(model='qwen3-8b',messages=solver_messages(prompt,problem),temperature=0.2,top_p=0.8,
        presence_penalty=0,frequency_penalty=0,max_tokens=capacity,
        extra_body=dict(enable_thinking=False,top_k=20,min_p=0))


def choose_membership(rows,excluded_ids,n=60,seed=81):
    if n!=60 or seed!=81:raise ValueError('CALIBRATION_FROZEN_SELECTION_POLICY_MISMATCH')
    if len({r['stable_example_id'] for r in rows})!=len(rows):raise ValueError('OPTIMIZE_DUPLICATE_IDS')
    pool=[r for r in rows if r['stable_example_id'] not in set(excluded_ids)]
    actual=min(n,len(pool));groups=defaultdict(list)
    for row in pool:groups[(subject(row),row['content']['level'])].append(row)
    counts=quotas({k:len(v) for k,v in groups.items()},actual) if actual else {}
    chosen=set()
    for key,group in groups.items():
        ordered=sorted(group,key=lambda r:(hashlib.sha256((PLAN_ID+'|'+str(seed)+'|'+r['stable_example_id']).encode()).hexdigest(),r['stable_example_id']))
        chosen.update(r['stable_example_id'] for r in ordered[:counts[key]])
    selected=[r for r in rows if r['stable_example_id'] in chosen]
    assert len(selected)==actual and not chosen&set(excluded_ids)
    return selected,dict(requested_n=n,available_optimize=len(rows),available_after_exclusion=len(pool),actual_n=actual,
        excluded_canary_ids=len(set(excluded_ids)),seed=seed,
        selection='SUBJECT_LEVEL_LARGEST_REMAINDER_LEXICAL_TIES_THEN_SHA256_PLAN_SEED_ID',
        output_order='FROZEN_OPTIMIZE_MEMBERSHIP_ORDER',uses_model_outcomes=False,
        subject_counts=dict(Counter(subject(r) for r in selected)),
        level_counts=dict(Counter(r['content']['level'] for r in selected)),
        population_strata={s+'|'+l:len(v) for (s,l),v in sorted(groups.items())},
        selected_strata={s+'|'+l:counts[(s,l)] for s,l in sorted(groups)},
        uncovered_population_strata=[s+'|'+l for s,l in sorted(groups) if not counts[(s,l)]])


def plan_artifact(selected,selection,source_identities):
    membership=[dict(example_id_sha256=hashlib.sha256(r['stable_example_id'].encode()).hexdigest(),
        content_sha256=r['content_sha256'],input_sha256=r['input_sha256'],
        reference_sha256=hashlib.sha256(r['reference_final_answer'].encode()).hexdigest(),
        subject=subject(r),level=r['content']['level']) for r in selected]
    plan=dict(schema_version='math_baseline_calibration_plan_v1',identity=PLAN_ID,
        scope='INDEPENDENT_BASELINE_DIAGNOSIS_NOT_PROMPT_OPTIMIZATION',status='DESIGN_FROZEN_NO_PAID_RUNNER',
        real_api_authorized=False,READY_TO_RUN=False,source_identities=source_identities,
        sampling=deepcopy(SAMPLING),model='qwen3-8b',solver_user_message='EXACT_ORIGINAL_PROBLEM',
        arms={a:dict(role=ROLE,strategy=STRATEGY,answer=ANSWERS[a],
            parser=NATIVE_ID if a in ('A','B') else BOXED_ID,
            rendered_system_sha256=SystemPrompt(ROLE,STRATEGY,ANSWERS[a]).system_hash) for a in ANSWERS},
        secondary=dict(identity=DIAGNOSTIC_ID,usage_scope='DIAGNOSTIC_ONLY',gold_blind=True,
            primary_override=False,never_search_gold_matching_intermediates=True),
        data=dict(role='optimize',selection=selection,membership=membership,membership_sha256=digest(membership),
            validation_raw_access=False,test_raw_access=False,shadow_raw_access=False,
            previously_adaptive_development_pool=True,independent_generalization_claim=False),
        retry=dict(successful_semantic_draws=4,trigger='NATIVE_OUTPUT_INVALID_ONLY_NEVER_MATH_INCORRECT',
            transport_max_retries=20,transport_attempts_per_draw=21,transport_failure_consumes_draw=False,
            first_capacity=3600,truncation_recovery_capacity=6144,
            expansion='ONLY_IMMEDIATELY_PREVIOUS_SUCCESS_FINISH_LENGTH',
            same_messages_and_sampling=True,incomplete_recovery='NOT_A_COMPLETED_LOGICAL_SCORE'),
        stages=dict(stage1=dict(members=1,n=selection['actual_n'],independent_scope=True),
            stage2=dict(members=5,n=selection['actual_n'],independent_scope=True,
                status='DESIGN_ONLY_SEPARATE_REVIEW_AND_AUTHORIZATION_REQUIRED',
                independent_realizations=True,stage1_response_reuse=False,
                aggregation='equal_equivalence_plurality_consistency_v1')),
        pairing=dict(unit='EXAMPLE_ID',within_arm_example_order='IDENTICAL',
            arm_order='ABC_BCA_CAB_CYCLIC_BY_EXAMPLE_INDEX',optimizer_concurrency=0,solver_concurrency=1,
            comparisons=['A_B_PROMPT_ONLY','A_C_OUTPUT_INTERFACE','B_C_OUTPUT_INTERFACE'],
            bootstrap_seed=81,bootstrap_resamples=10000,bootstrap_unit='PAIRED_EXAMPLE_NOT_MEMBER',
            inference='SMALL_DEVELOPMENT_DIAGNOSIS_NO_POPULATION_SIGNIFICANCE_CLAIM'),
        metrics=dict(primary=['FIRST_DRAW_FORMAT_VALID','FIRST_DRAW_NATIVE_CORRECT',
            'RECOVERED_NATIVE_VALID','RECOVERED_NATIVE_CORRECT','TOKENS_PER_LOGICAL','COST_PER_CORRECT'],
            secondary=['COMMON_DIAGNOSTIC_STATUS','COMMON_DIAGNOSTIC_CORRECT','ERROR_CATEGORY',
                'TRUNCATION','PAIRED_IMPROVED_REGRESSED','SUBJECT_LEVEL']),
        disabled_components=['responsibility','gradient','pattern','layer1','team_probe','full_commit','shadow_writeback','memory'],
        budget_policy='FRESH_STAGE_SPECIFIC_CHARGED_PLUS_RESERVED_CAP_NO_OLD_BALANCE',
        stopping='NATIVE_VALID_OR_FOUR_INVALID_PER_LOGICAL_FIXED_N_OR_OPERATIONAL_FAIL_CLOSED',
        incomplete_stage='PUBLISH_FACTUAL_PARTIAL_COUNTERS_ONLY_NO_ARM_EFFICACY_COMPARISON',
        future_authorization_required=['new_source_commit','protocol_manifest_membership_hashes','models_and_provider_policy',
            'stage_roles_seed_order','native_parsers_and_common_diagnostic','new_cache_namespace_and_accounting_scope',
            'charged_plus_reserved_ceiling','retry_capacity_policy','no_validation_test_access','single_use_exact_scope'],
        real_execution_implementation='NOT_PRESENT_FAKE_AND_DRY_RUN_ONLY')
    plan['protocol_sha256']=digest(plan)
    return plan


def validate_plan(plan):
    value={k:v for k,v in plan.items() if k!='protocol_sha256'}
    if digest(value)!=plan.get('protocol_sha256'):raise ValueError('CALIBRATION_PROTOCOL_HASH_MISMATCH')
    if (plan.get('identity')!=PLAN_ID or plan.get('sampling')!=SAMPLING
            or plan.get('real_api_authorized') is not False or plan.get('READY_TO_RUN') is not False
            or plan.get('model')!='qwen3-8b' or plan['data']['role']!='optimize'
            or any(plan['data'][k] for k in ('validation_raw_access','test_raw_access','shadow_raw_access'))):
        raise ValueError('CALIBRATION_SCOPE_MISMATCH')
    for arm in ANSWERS:
        expected=dict(role=ROLE,strategy=STRATEGY,answer=ANSWERS[arm],parser=NATIVE_ID if arm in ('A','B') else BOXED_ID,
            rendered_system_sha256=SystemPrompt(ROLE,STRATEGY,ANSWERS[arm]).system_hash)
        if plan['arms'][arm]!=expected:raise ValueError('CALIBRATION_ARM_MISMATCH')
    members=plan['data']['membership']
    if (digest(members)!=plan['data']['membership_sha256']
            or len({r['example_id_sha256'] for r in members})!=len(members)):
        raise ValueError('CALIBRATION_MEMBERSHIP_MISMATCH')


def budget_artifact(plan,selected,historical):
    validate_plan(plan);n=len(selected);result={}
    # Exact UTF-8 byte + 4096 reservation matches the frozen existing transport
    # accounting convention. It is a reservation, not a tokenizer measurement.
    inputs={a:[len(serialized_request(request_for(a,r['content']['problem'],6144)))+4096 for r in selected] for a in ANSWERS}
    observed=historical['reported_success_tokens']/historical['logical_examples']
    timeout=historical['conservative_initial_timeout_charge']/historical['logical_examples']
    for stage,members in (('stage1',1),('stage2',5)):
        logical=3*n*members;semantic=4*logical;transport=21*semantic
        worst_input=members*4*sum(sum(v) for v in inputs.values())
        output=logical*(3600+3*6144)
        tolerance=0  # Existing Solver accounting has no optimizer tolerance.
        history_scenario=logical*(observed+timeout)
        # Budget choice is independently derived from this calibration panel,
        # with a 2x observed-rate allowance and prompt/input reservation margin.
        input_margin=members*4*sum(sum(max(0,v-a) for v,a in zip(inputs[arm],inputs['A'],strict=True)) for arm in ('B','C'))
        proposed_cap=math.ceil((2*history_scenario+input_margin)/10000)*10000
        result[stage]=dict(logical_requests=logical,logical_requests_per_arm=n*members,
            successful_semantic_responses_upper=semantic,transport_attempts_upper=transport,
            transport_retry_extra_attempts_upper=20*semantic,
            first_draw_output_tokens_upper=logical*3600,
            successful_recovered_output_tokens_upper=output,
            successful_input_reservation_upper=worst_input,
            output_measurement_tolerance_upper=tolerance,
            no_transport_failure_reserved_envelope=worst_input+output+tolerance,
            all_transport_attempts_conservatively_charged_envelope=21*(worst_input+output+tolerance),
            historical_rate_scenario_reported_tokens=logical*observed,
            historical_rate_scenario_successful_responses=logical*historical['successful_responses']/historical['logical_examples'],
            historical_rate_scenario_transport_timeouts=logical*historical['transport_timeouts']/historical['logical_examples'],
            historical_rate_scenario_physical_attempts=logical*(historical['successful_responses']+historical['transport_timeouts'])/historical['logical_examples'],
            historical_rate_scenario_input_tokens=logical*sum(r['input_tokens'] for r in historical['rounds'])/historical['logical_examples'],
            historical_rate_scenario_output_tokens=logical*sum(r['output_tokens'] for r in historical['rounds'])/historical['logical_examples'],
            historical_rate_scenario_unknown_usage_charge=logical*timeout,
            historical_rate_scenario_charged_tokens=history_scenario,
            historical_first_draw_scenario_tokens=logical*historical['rounds'][0]['reported_tokens']/historical['logical_examples'],
            prompt_input_reservation_margin=input_margin,
            proposed_independent_charged_plus_reserved_cap=proposed_cap,
            cap_derivation='CEIL_TO_10K(2 * HISTORICAL_CHARGED_RATE * LOGICAL_COUNT + POSITIVE_B_AND_C_MINUS_A_INPUT_MARGIN)',
            saturation_or_complete_panel_guaranteed=False,
            assumption='HISTORICAL_RATE_SCENARIO_NOT_A_FORECAST_FOR_NEW_PROMPTS_OR_60_EXAMPLES',
            authorization='NONE_FRESH_EXACT_STAGE_SCOPE_REQUIRED')
    return dict(identity='MATH_BASELINE_CALIBRATION_BUDGET_V1',protocol_sha256=plan['protocol_sha256'],
        successful_draws_per_logical_upper=4,transport_attempts_per_successful_draw_upper=21,
        capacity_sequence_upper=[3600,6144,6144,6144],
        input_bound='SERIALIZED_UTF8_BYTES_PLUS_4096_NOT_TOKENIZER_MEASUREMENT',
        input_reservation_by_arm={a:dict(min=min(v) if v else 0,max=max(v) if v else 0,mean=sum(v)/n if n else 0) for a,v in inputs.items()},
        historical_rate_scenario_source='EXISTING_V25_INITIAL_60_LOGICAL_192_SUCCESS_2_TIMEOUT',
        model_price='UNKNOWN_NO_NEW_PRICING_OR_BILLING_QUERY_CURRENCY_COST_NOT_ESTIMATED',
        old_2m_scope_reusable=False,stages=result,real_api_calls=0)


class FakeProvider:
    """Closed predetermined fixture responses, with no transport/client port."""
    def __init__(self,responses):self.responses=deepcopy(responses);self.calls=[]
    def complete(self,arm,example_id,member,ordinal,request,physical=1,stage='stage1'):
        self.calls.append(dict(arm=arm,example_id=example_id,member=member,ordinal=ordinal,
            physical_attempt=physical,stage=stage,request=deepcopy(request),
            realization_identity=digest(dict(plan=PLAN_ID,stage=stage,arm=arm,
                example_id=example_id,member=member,draw=ordinal,physical=physical))))
        fixture=self.responses[(arm,example_id,member)][ordinal-1]
        if isinstance(fixture,list):fixture=fixture[physical-1]
        return deepcopy(fixture)


def fake_recovery(arm,example,member,provider,stage='stage1'):
    if type(provider) is not FakeProvider:raise ValueError('FAKE_PROVIDER_REQUIRED_NO_REAL_EXECUTION_PORT')
    if stage not in {'stage1','stage2'} or (stage=='stage1' and member!=0) or not 0<=member<5:
        raise ValueError('FAKE_STAGE_MEMBER_MISMATCH')
    attempts=[];previous=None;transport_retries=0
    for ordinal in range(1,5):
        capacity=6144 if previous in {'length','max_tokens','max_output_tokens'} else 3600
        request=request_for(arm,example['problem'],capacity)
        for physical in range(1,22):
            response=provider.complete(arm,example['example_id'],member,ordinal,request,physical,stage)
            if 'error' not in response:break
            if response['error']!='TIMEOUT' or physical==21:
                raise ValueError('FAKE_TRANSPORT_ABORT_INCOMPLETE_LOGICAL')
            transport_retries+=1
        native=native_extract(arm,response['text'],response.get('finish_reason','stop'))
        secondary=diagnostic_extract(response['text'],response.get('finish_reason','stop'))
        attempts.append(dict(semantic_attempt_index=ordinal,output_capacity=capacity,native=native,
            primary_correct=primary_correctness(native,example['reference']),
            secondary_status=secondary.status,secondary_correct=grade_diagnostic(secondary,example['reference'])))
        previous=response.get('finish_reason','stop')
        if native['valid']:break
    return dict(first_valid=attempts[0]['native']['valid'],first_correct=attempts[0]['primary_correct'],
        recovered_valid=attempts[-1]['native']['valid'],recovered_correct=attempts[-1]['primary_correct'],
        semantic_draws=len(attempts),transport_retries=transport_retries,attempts=attempts)
