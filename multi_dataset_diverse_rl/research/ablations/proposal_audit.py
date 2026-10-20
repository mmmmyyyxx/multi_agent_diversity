"""Streaming single-Parent proposal lineage, evidence, gate and cost audit."""
from copy import deepcopy

from ...benchmarks.data_freeze import digest
from ...search.system_prompt import SystemPrompt,candidate_failed_checks
from ...persistence.durable_io import read_json
from .interfaces import PreparedFeedback
from .pattern_proposals import require,project_proposals,winner,score,pair_changes,current_view,repair_metrics
from .proposal_config import slots


def sealed(value):
    require(value.get('integrity_sha256')==digest({k:v for k,v in value.items() if k!='integrity_sha256'}),'CHECKPOINT_SEAL')


def profile(value,ids):
    require(isinstance(value,dict) and set(value)==set(ids) and all(type(v.get('correct')) is bool
        and type(v.get('valid')) is bool and (not v['correct'] or v['valid']) for v in value.values()),'PROFILE_MEMBERSHIP_VALIDITY')


def audit_initial(cp,scope,oi,vi,*,check_seal=True,check_initial_prompt=True):
    if check_seal:sealed(cp)
    require(cp['scope']==scope and cp['scope_identity']==digest(scope),'SCOPE')
    require(scope['optimize_membership_sha256']==digest(list(oi)) and scope['validation_membership_sha256']==digest(list(vi)),'SCOPE_MEMBERSHIP')
    if check_initial_prompt:
        require(SystemPrompt.from_dict(cp['prompt']).prompt_hash==scope['initial_prompt_sha256'],'INITIAL_PROMPT')
    profile(cp['optimize'],oi);profile(cp['validation'],vi)


def audit_transition(previous,cp,plan,cell,n,read,*,optimize_ids=None):
    from .generation import candidate_checks
    sealed(cp);parent=SystemPrompt.from_dict(previous['prompt'])
    require(cp['scope_identity']==previous['scope_identity'] and cp['parent_prompt']==parent.to_dict(),'PARENT')
    profile(cp['optimize'],previous['optimize']);profile(cp['validation'],previous['validation'])
    from .full_error_control import SCHEMA as control_schema,audit_raw_module
    raw_control=plan['config']['schema_version']==control_schema
    module=read(f'round_{n}_module');audit=module['audit']
    full=audit['raw_feedback'] if raw_control else audit['complete_diagnosis']
    wrong={i for i,v in previous['optimize'].items() if not v['correct']}
    require(module['module_binding']==plan['module_bindings'][cell['arm_id']] and module['module_state'] is None,'MODULE_BINDING')
    require({e['example_id'] for e in full['errors']}==wrong and len(full['errors'])==len(wrong),'DIAGNOSIS_UNIVERSE')
    if raw_control:
        audit_raw_module(audit,previous,cell,list(previous['optimize']) if optimize_ids is None else optimize_ids,
                         cp['metric']['shared_diagnostic_cost'])
    else:
        require({d['example_id'] for d in full['structured_feedback']['diagnostics']}==wrong,'DIAGNOSIS_UNIVERSE')
        require(audit['sampling_context']['prompt_sha256']==parent.prompt_hash
            and audit['sampling_context']['cell_id']==cell['cell_id'],'SAMPLING_PARENT')
    for error in full['errors']:
        observed=previous['optimize'][error['example_id']]
        require(error['correct'] is False and error['valid']==observed['valid']
            and error['prediction']==observed['answer'],'DIAGNOSIS_OBSERVATION')
    views,selection=project_proposals(PreparedFeedback([],None,None,audit,None),42,n,cell['proposal_mode'],raw_control=raw_control)
    require(module['selection']==selection,'SELECTION')
    m=cp['metric'];proposals=m['proposals']
    require(m['round']==n and m['validation_before']==score(previous['validation'])
        and m['optimize_before']==score(previous['optimize'])
        and len(proposals)==len(views) and all(m[k]==v for k,v in selection.items()),'METRIC_SELECTION')
    costs=deepcopy(m['shared_diagnostic_cost'])
    sequential=cell['proposal_mode']=='sequential'
    actual_parent=parent;actual_optimize=previous['optimize'];actual_validation=previous['validation']
    expected_acceptances=[]
    for slot,(p,view) in enumerate(zip(proposals,views)):
        view,diagnosis_context,eligible=current_view(PreparedFeedback([],None,None,audit,None),view,parent,
            actual_parent,actual_optimize,42,n)
        feedback=view['feedback'];op=read(f'round_{n}_proposal_{slot}_opportunity')
        ident=f"{cell['cell_id']}:round{n}:proposal{slot}"
        expected_op=dict(parent_prompt=actual_parent.to_dict(),errors=feedback['errors'],structured_feedback=feedback['structured_feedback'],
            module_context=None,retained_validation_correct=score(actual_validation),proposal_id=ident,focus=view['focus'])
        if diagnosis_context is not None:expected_op['diagnosis_context']=diagnosis_context
        require(op==expected_op,'EVIDENCE_ISOLATION')
        require(p['slot']==slot and p['proposal_id']==ident and p['parent_prompt_sha256']==actual_parent.prompt_hash
            and p['parent_prompt']==actual_parent.to_dict() and p['diagnosis_source_prompt_sha256']==parent.prompt_hash
            and p['validation_before']==score(actual_validation) and p['optimize_before']==score(actual_optimize)
            and p['current_focus']==view['focus']
            and p['proposal_pattern_id']==selection['selected_pattern_ids'][slot]
            and p==read(f'round_{n}_proposal_{slot}_result'),'PROPOSAL_IDENTITY')
        require(type(p['generation_draws']) is int and (1<=p['generation_draws']<=4 if eligible else p['generation_draws']==0),'DRAW_COUNT')
        for draw in range(1,p['generation_draws']+1):
            g=read(f'round_{n}_proposal_{slot}_generation_{draw}')
            require(g['draw']==draw and isinstance(g['request_sha256'],str) and len(g['request_sha256'])==64,'GENERATION_RECEIPT_REFERENCE')
            if draw<p['generation_draws']:
                require(g['errors'] and g['candidate'] is None and g['disposition']=='FORMAT_RETRY','RETRY_CHAIN')
        child=p['candidate_prompt']
        if eligible:require(g['candidate']==child,'GENERATION_CANDIDATE')
        if not eligible:
            require(child is None and p['generation_status']=='SKIPPED_NO_CURRENT_ERRORS'
                and all(v==0 for v in p['generation_cost'].values()),'EMPTY_CURRENT_TARGET')
        elif child is None:
            require(p['generation_status']=='EXHAUSTED' and p['generation_draws']==4
                and g['errors'] and g['disposition']=='FORMAT_RETRY','EXHAUSTION')
        else:
            candidate=SystemPrompt.from_dict(child)
            require(not candidate_checks(candidate,parent=actual_parent,errors=feedback['errors']) and not g['errors'],'CANDIDATE_STRUCTURE')
            if child==actual_parent.to_dict():
                require(p['generation_status']=='NO_OP' and g['disposition']=='NO_OP','NO_OP')
            else:
                require(p['generation_status']=='MEASURED' and g['disposition']=='GENERATED','MEASURED_STATUS')
                profile(p['candidate_optimize'],previous['optimize']);profile(p['candidate_validation'],previous['validation'])
                require(p['candidate_optimize_score']==score(p['candidate_optimize'])
                    and p['candidate_validation_score']==score(p['candidate_validation']),'SCORES')
                require(p['repair_metrics']==repair_metrics(previous['optimize'],actual_optimize,
                    p['candidate_optimize'],view['focus']['selected_support_ids']),'REPAIR_METRICS')
        from ...search.system_prompt import block_lineage
        require(p['lineage']==(None if child is None else block_lineage(actual_parent,SystemPrompt.from_dict(child))),'BLOCK_LINEAGE')
        same=(p['lineage']['edited_blocks']==proposals[0]['lineage']['edited_blocks']) if (
            sequential and slot and proposals[0]['accepted'] and p['generation_status']=='MEASURED') else None
        require(p['same_block_as_first_accepted_edit']==same,'OVERWRITTEN_BLOCK_RECORD')
        if p['generation_status']!='MEASURED':
            require(all(p[k] is None for k in ('candidate_optimize','candidate_validation','candidate_optimize_score','candidate_validation_score'))
                and all(v==0 for v in p['evaluation_cost'].values()),'UNMEASURED_PROFILE')
            require(p['repair_metrics'] is None,'UNMEASURED_REPAIR')
        for cost in (p['generation_cost'],p['evaluation_cost']):
            require(set(cost)==set(costs) and all(type(v) is int and v>=0 for v in cost.values()),'COST_DOMAIN')
            costs={k:costs[k]+cost[k] for k in costs}
        if sequential:
            accepted=p['generation_status']=='MEASURED' and p['candidate_validation_score']>score(actual_validation)
            require(p['accepted'] is accepted,'SEQUENTIAL_LATEST_PARENT_GATE')
            if accepted:
                expected_acceptances.append(p['proposal_id']);actual_parent=SystemPrompt.from_dict(child)
                actual_optimize,actual_validation=p['candidate_optimize'],p['candidate_validation']
    chosen=None if sequential else winner(proposals,score(previous['validation']))
    if not sequential:
        for p in proposals:require(p['accepted'] is (chosen is not None and p['proposal_id']==chosen['proposal_id']),'PARALLEL_WINNER')
        if chosen is not None:
            expected_acceptances=[chosen['proposal_id']];actual_parent=SystemPrompt.from_dict(chosen['candidate_prompt'])
            actual_optimize,actual_validation=chosen['candidate_optimize'],chosen['candidate_validation']
    for p in proposals:
        reason='ACCEPTED_STRICT_VALIDATION_GAIN' if p['accepted'] else (
            ('REJECTED_PARALLEL_COMPETITION' if not sequential and p['candidate_validation_score']>p['validation_before']
                else 'REJECTED_NO_STRICT_VALIDATION_GAIN') if p['generation_status']=='MEASURED' else p['generation_status'])
        require(p['decision']==reason and p['generation_seconds']>=0 and p['evaluation_seconds']>=0,'PROPOSAL_DECISION_OR_TIME')
    measured=any(p['generation_status']=='MEASURED' for p in proposals)
    decision='ACCEPTED_STRICT_VALIDATION_GAIN' if expected_acceptances else 'REJECTED_NO_STRICT_VALIDATION_GAIN' if measured else (
        'SKIPPED_NO_OP' if any(p['generation_status']=='NO_OP' for p in proposals) else 'ABANDONED_GENERATION_RETRIES_EXHAUSTED')
    require(m['accepted'] is bool(expected_acceptances) and m['accepted_proposal_ids']==expected_acceptances
        and m['accepted_update_count']==len(expected_acceptances)
        and m['winning_proposal_id']==(None if chosen is None else chosen['proposal_id'])
        and m['decision']==decision and m['candidate_evaluated']==measured,'WINNER_STRICT_GATE')
    require(m['distinct_pattern_count']==len({p['proposal_pattern_id'] for p in proposals if p['generation_draws']>0 and p['proposal_pattern_id'] is not None})
        and m['evaluated_distinct_pattern_count']==len({p['proposal_pattern_id'] for p in proposals if p['generation_status']=='MEASURED' and p['proposal_pattern_id'] is not None}),'ACTUAL_DIRECTION_COUNT')
    require(m['cost']==costs and m['pair_changes']==(dict(status='SEQUENTIAL_DIFFERENT_PARENTS_NOT_SAME_PARENT_COMPARISON')
        if sequential else pair_changes(previous['optimize'],proposals,views)),'COST_OR_PAIR_ANALYSIS')
    require(cp['prompt']==actual_parent.to_dict() and cp['optimize']==actual_optimize and cp['validation']==actual_validation,'RETAINED_STATE')
    require(cp['module_state'] is None,'UNEXPECTED_MEMORY')


def audit_cells(directory,scope,results=None,allow_partial=False,*,cell_offset=0):
    from .config import validate_plan,request_namespace
    plan=scope['plan'];validate_plan(plan)
    results=read_json(directory/'results.json') if results is None else results
    require(type(cell_offset) is int and 0<=cell_offset<len(plan['cells']) and (allow_partial or cell_offset==0),'AUDIT_CELL_OFFSET')
    expected=plan['cells'][cell_offset:cell_offset+len(results)] if allow_partial else plan['cells']
    require([r['cell'] for r in results]==expected,'COMPLETE_ORDERED_CELLS')
    for cell,result in zip(expected,results):
        folder=directory/cell['cell_id'];read=lambda name:read_json(folder/(name+'.json'))
        previous=read('checkpoint_0');binding=plan['module_bindings'][cell['arm_id']]
        target=dict(plan_identity=plan['plan_identity'],cell=cell,initial_prompt_sha256=scope['initial_prompt_sha256'],
            optimize_membership_sha256=digest(scope['optimize_sample_ids']),
            validation_membership_sha256=scope['gate_membership']['membership_sha256'],module_binding=binding,
            repeat_namespace=request_namespace(plan['plan_identity'],42,cell['repeat_id']))
        audit_initial(previous,target,scope['optimize_sample_ids'],scope['gate_membership']['sample_ids'])
        count=result['rounds_completed'];require(0<=count<=plan['config']['rounds'],'ROUND_COUNT')
        require(allow_partial or count==plan['config']['rounds'],'INCOMPLETE_TRAINING')
        history=[previous['prompt']];metrics=[];usage=deepcopy(previous['bootstrap_usage'])
        for n in range(1,count+1):
            current=read(f'checkpoint_{n}');audit_transition(previous,current,plan,cell,n,read,optimize_ids=scope['optimize_sample_ids'])
            history.append(current['prompt']);metrics.append(current['metric'])
            usage={k:usage[k]+current['metric']['cost'][k] for k in usage};previous=current
        require(result['prompt_history']==history and result['metrics']==metrics and result['final_prompt']==previous['prompt']
            and result['scope_identity']==digest(target) and result['module_binding']==binding
            and result['final_optimize_correct']==score(previous['optimize'])
            and result['final_validation_correct']==score(previous['validation'])
            and result['optimize_count']==len(scope['optimize_sample_ids'])
            and result['validation_count']==len(scope['gate_membership']['sample_ids']) and result['usage']==usage,'FINAL_RESULT')
        if not allow_partial or (folder/'result.json').exists():require(result==read('result'),'PERSISTED_RESULT')
    return results
