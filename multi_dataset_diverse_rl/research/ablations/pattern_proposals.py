"""Single frozen Parent, one diagnosis, isolated proposals and measured winner."""
from copy import deepcopy
from time import perf_counter

from ...benchmarks.data_freeze import digest
from ...search.schemas import SearchContractError
from ...search.system_prompt import SystemPrompt,block_lineage
from ...search.pattern_selection import LargestUniqueSupportSelector
from .full_pattern import project_full
from .interfaces import PreparedFeedback,OptimizeContext,ModuleProvider
from .generation import generate_candidate
from .proposal_config import slots


def require(value,reason):
    if not value:raise SearchContractError('PROPOSAL_'+reason)


def project_proposals(prepared,seed,ordinal,mode,*,raw_control=False):
    """Reproject the second focus with the same native sampler, never rediagnose."""
    if raw_control:
        from .full_error_control import project_feedback
        require(mode=='parallel','RAW_CONTROL_MODE')
        return project_feedback(prepared.audit)
    full=prepared.audit['complete_diagnosis'];sampling=prepared.audit['sampling_context']
    patterns=full['structured_feedback']['patterns']
    require(len({p['pattern_id'] for p in patterns})==len(patterns),'DUPLICATE_PATTERN')
    first,focus=project_full(full,seed,ordinal,sampling)
    views=[dict(feedback=first,focus=focus)]
    if slots(mode)==2 and len(patterns)>1:
        second=deepcopy(views[0])
        if len(patterns)>1:
            remaining=[p for p in patterns if p['pattern_id']!=focus['selected_pattern_id']]
            chosen,_=LargestUniqueSupportSelector().select(remaining,seed=seed,ordinal=ordinal)
            isolated=deepcopy(full)
            isolated['structured_feedback']['patterns']=[chosen]
            feedback,second_focus=project_full(isolated,seed,ordinal,sampling)
            second=dict(feedback=feedback,focus=second_focus)
        views.append(second)
    ids=[v['focus']['selected_pattern_id'] for v in views]
    return views,dict(pattern_count=len(patterns),selected_pattern_ids=ids,
        pattern_support_counts=[len(v['focus']['selected_support_ids']) for v in views],
        selected_distinct_pattern_count=len({i for i in ids if i is not None}),
        degeneration='NO_PATTERN_ONE_RAW_BATCH' if not patterns else 'ONE_PATTERN_ONE_PROPOSAL' if len(patterns)==1 else None)


def winner(proposals,before):
    measured=[p for p in proposals if p['generation_status']=='MEASURED']
    if not measured:return None
    best=min(measured,key=lambda p:(-p['candidate_validation_score'],p['slot']))
    return best if best['candidate_validation_score']>before else None


def delta(after,before):return {k:after[k]-v for k,v in before.items()}
def score(profile):return sum(v['correct'] for v in profile.values())


def current_view(prepared,view,diagnosis_parent,current_parent,optimize,seed,ordinal):
    """Keep the root Pattern identity; resample only still-wrong support evidence."""
    view=deepcopy(view);support=view['focus']['selected_support_ids']
    if view['focus'].get('raw_all_error_control'):
        require(diagnosis_parent==current_parent,'RAW_CONTROL_PARENT_CHANGED')
        return view,None,True
    active=[i for i in support if not optimize[i]['correct']]
    view['focus']['active_support_ids']=active
    if support and not active:return view,None,False
    if diagnosis_parent==current_parent:return view,None,True
    require(bool(support),'UNEXPECTED_FALLBACK_AFTER_UPDATE')
    full=deepcopy(prepared.audit['complete_diagnosis'])
    original=view['feedback']['structured_feedback']['patterns'][0]
    full['structured_feedback']['patterns']=[dict(original,support_ids=active)]
    feedback,focus=project_full(full,seed,ordinal,prepared.audit['sampling_context'])
    feedback['structured_feedback']['patterns']=[deepcopy(original)]
    view['feedback']=feedback
    view['focus'].update(representative_example_ids=focus['representative_example_ids'],
        representative_sampling_support_ids=active)
    from ...benchmarks.math_response_evidence import adaptive_trajectory
    from ...search.group_diagnosis import readable_observation
    observations=[]
    for error in feedback['errors']:
        i=error['example_id'];observed=optimize[i]
        trajectory=adaptive_trajectory(observed['profile'],member_id=0,prompt=current_parent,
            example_id=i,problem=error['problem'])
        observations.append(dict(example_id=i,prompt_sha256=current_parent.prompt_hash,
            **readable_observation(trajectory,correct=observed['correct'])))
    context=dict(diagnosis_source_prompt_sha256=diagnosis_parent.prompt_hash,
        current_parent_prompt_sha256=current_parent.prompt_hash,original_support_ids=support,
        active_support_ids=active,current_parent_observations=observations,
        diagnosis_status='frozen_round_start_diagnosis_not_current_parent_diagnosis')
    return view,context,True


def repair_metrics(root_profile,before,after,support):
    support=set(support);wrong={i for i,v in before.items() if not v['correct']}
    prior={i for i,v in root_profile.items() if not v['correct'] and before[i]['correct']}
    return dict(support_count=len(support),support_errors_before=len(support&wrong),
        support_repairs=sum(after[i]['correct'] for i in support&wrong),
        support_regressions=sum(before[i]['correct'] and not after[i]['correct'] for i in support),
        support_errors_remaining=sum(not after[i]['correct'] for i in support),
        outside_support_repairs=sum(after[i]['correct'] for i in wrong-support),
        outside_support_regressions=sum(before[i]['correct'] and not after[i]['correct'] for i in set(before)-support),
        prior_root_repairs=len(prior),preserved_prior_root_repairs=sum(after[i]['correct'] for i in prior),
        broken_prior_root_repairs=sum(not after[i]['correct'] for i in prior))


def pair_changes(parent,proposals,views):
    if len(proposals)!=2 or any(p['generation_status']!='MEASURED' for p in proposals):
        return dict(status='NOT_TWO_MEASURED_PROPOSALS')
    a,b=[p['candidate_optimize'] for p in proposals]
    wrong={i for i,v in parent.items() if not v['correct']}
    return dict(status='BOTH_MEASURED',
        optimize_correctness_disagreement=sum(a[i]['correct']!=b[i]['correct'] for i in a),
        first_only_repairs=sum(a[i]['correct'] and not b[i]['correct'] for i in wrong),
        second_only_repairs=sum(b[i]['correct'] and not a[i]['correct'] for i in wrong),
        second_only_repairs_by_selected_support=[sum(i in wrong and b[i]['correct'] and not a[i]['correct']
            for i in v['focus']['selected_support_ids']) for v in views],
        first_regressions=sum(v['correct'] and not a[i]['correct'] for i,v in parent.items()),
        second_regressions=sum(v['correct'] and not b[i]['correct'] for i,v in parent.items()))


def run_cell(plan,cell,initial_prompt,optimize_rows,validation_rows,broker,persist,*,registry=None,resume=None):
    from .config import validate_plan,request_namespace
    from .modules import default_registry
    from .engine import _observations,_checkpoint
    from ..single_prompt import raw_error
    from .proposal_audit import audit_transition,audit_initial
    from .full_error_control import SCHEMA as control_schema,bind_feedback
    raw_control=plan['config']['schema_version']==control_schema
    registry=registry or default_registry();validate_plan(plan,registry)
    require(cell in plan['cells'] and broker.phase=='train','CELL_PHASE')
    optimize_rows,validation_rows=tuple(optimize_rows),tuple(validation_rows)
    oi=[r['stable_example_id'] for r in optimize_rows];vi=[r['stable_example_id'] for r in validation_rows]
    require(oi and vi and len(oi)==len(set(oi)) and len(vi)==len(set(vi)) and not set(oi)&set(vi),'SPLIT_MEMBERSHIP')
    broker.bind_cell(cell,plan);begin=broker.usage_view();initial=SystemPrompt.from_dict(initial_prompt)
    binding=plan['module_bindings'][cell['arm_id']];module=registry.create(cell['module'])
    scope=dict(plan_identity=plan['plan_identity'],cell=cell,initial_prompt_sha256=initial.prompt_hash,
        optimize_membership_sha256=digest(oi),validation_membership_sha256=digest(vi),module_binding=binding,
        repeat_namespace=request_namespace(plan['plan_identity'],cell['seed'],cell['repeat_id']))
    scope_id=digest(scope);metrics=[];history=[];old_usage={k:0 for k in begin}
    partial={} if resume is None else resume.get('partial',{})
    if partial:
        from .proposal_continuation import validate_partial_resume
        validate_partial_resume(resume,cell)
        original_persist=persist
        def persist(name,data):
            if name in resume['artifacts']:
                require(data==resume['artifacts'][name],'RESTORED_ARTIFACT_CHANGED')
            else:original_persist(name,data)
    if resume is None:
        bootstrap_clock=perf_counter()
        optimize=_observations(optimize_rows,broker.evaluate(initial,optimize_rows,'optimize',42,0))
        validation=_observations(validation_rows,broker.evaluate(initial,validation_rows,'validation',42,0))
        previous=dict(scope_identity=scope_id,scope=scope,prompt=initial.to_dict(),
            optimize=optimize,validation=validation,module_state=None,bootstrap_usage=delta(broker.usage_view(),begin),
            bootstrap_wall_seconds=perf_counter()-bootstrap_clock)
        previous=dict(previous,integrity_sha256=digest(previous));persist('checkpoint_0',previous)
        history=[initial.to_dict()];bootstrap_seconds=previous['bootstrap_wall_seconds']
    else:
        artifacts=resume['artifacts'];previous=None
        for n,cp in enumerate(resume['checkpoints']):
            require(n<=plan['config']['rounds'],'RESUME_TOO_LONG')
            if n==0:
                audit_initial(cp,scope,oi,vi)
                old_usage=deepcopy(cp['bootstrap_usage'])
                bootstrap_seconds=cp['bootstrap_wall_seconds']
            else:
                audit_transition(previous,cp,plan,cell,n,lambda name:artifacts[name])
                metrics.append(deepcopy(cp['metric']))
                old_usage={k:old_usage[k]+cp['metric']['cost'][k] for k in old_usage}
            history.append(deepcopy(cp['prompt']));previous=cp
        require(previous is not None and old_usage==resume['historical_usage'],'RESUME_USAGE')
        # A paid partial round cannot be silently regenerated. A future continuation
        # must bind exact receipts; completed boundaries never reissue Teacher calls.
        require(partial or not any(k.startswith(f'round_{len(metrics)+1}_') for k in artifacts),'PARTIAL_ROUND_REQUIRES_FRESH_CONTINUATION')
        if partial:old_usage={k:old_usage[k]+partial['usage'][k] for k in old_usage}
    audit_initial(dict(scope=scope,scope_identity=scope_id,prompt=initial.to_dict(),optimize=previous['optimize'],
        validation=previous['validation']),scope,oi,vi,check_seal=False,check_initial_prompt=False)
    for ordinal in range(len(metrics)+1,plan['config']['rounds']+1):
        parent=SystemPrompt.from_dict(previous['prompt']);optimize=previous['optimize'];validation=previous['validation']
        broker.round_index=ordinal;start=broker.usage_view();round_start=perf_counter()
        restoring=bool(partial and ordinal==partial['ordinal'])
        if restoring:start={k:v-partial['usage'][k] for k,v in start.items()}
        wrong=[r for r in optimize_rows if not optimize[r['stable_example_id']]['correct']]
        context=OptimizeContext(cell['cell_id'],42,cell['repeat_id'],ordinal,parent,deepcopy(tuple(wrong)),
            {r['stable_example_id']:deepcopy(optimize[r['stable_example_id']]) for r in wrong})
        provider=ModuleProvider(broker,binding['semantic_calls_per_opportunity'])
        if restoring:
            saved=artifacts[f'round_{ordinal}_module']
            prepared=PreparedFeedback([],None,None,deepcopy(saved['audit']),None)
        else:prepared=module.prepare(context,None,provider)
        require(isinstance(prepared,PreparedFeedback) and prepared.state is None and prepared.module_context is None,'MODULE_FEEDBACK')
        originals={r['stable_example_id']:raw_error(r,optimize[r['stable_example_id']]) for r in wrong}
        if raw_control:
            require(not restoring,'RAW_CONTROL_FRESH_ONLY')
            prepared=bind_feedback(prepared,parent,cell,previous,oi)
        else:require(prepared.audit['complete_diagnosis']['errors']==list(originals.values()),'DIAGNOSIS_SOURCE')
        views,selection=project_proposals(prepared,42,ordinal,cell['proposal_mode'],raw_control=raw_control)
        diagnostic_cost=deepcopy(partial['diagnostic_usage']) if restoring else delta(broker.usage_view(),start)
        record=dict(module_binding=binding,module_state=None,audit=dict(prepared.audit,
            diagnostic_draw_counts=(saved['audit']['diagnostic_draw_counts'] if restoring else provider.draw_counts())),selection=selection)
        persist(f'round_{ordinal}_module',record)
        proposals=[];sequential=cell['proposal_mode']=='sequential'
        current_parent,current_optimize,current_validation=parent,optimize,validation

        def generate(slot,view,actual_parent,actual_optimize,actual_validation):
            view,diagnosis_context,eligible=current_view(prepared,view,parent,actual_parent,actual_optimize,42,ordinal)
            proposal_id=f"{cell['cell_id']}:round{ordinal}:proposal{slot}";feedback=view['feedback']
            opportunity=dict(parent_prompt=actual_parent.to_dict(),errors=feedback['errors'],
                structured_feedback=feedback['structured_feedback'],module_context=None,
                retained_validation_correct=score(actual_validation),proposal_id=proposal_id,focus=view['focus'])
            if diagnosis_context is not None:opportunity['diagnosis_context']=diagnosis_context
            persist(f'round_{ordinal}_proposal_{slot}_opportunity',opportunity)
            before=broker.usage_view();clock=perf_counter();candidate=None;draws=0
            saved_key=f'round_{ordinal}_proposal_{slot}_generation_'
            saved_draws=sorted((v for k,v in artifacts.items() if k.startswith(saved_key)),key=lambda v:v['draw']) if restoring else []
            if saved_draws:
                final=saved_draws[-1];draws=final['draw']
                candidate=None if final['candidate'] is None else SystemPrompt.from_dict(final['candidate'])
            elif eligible:
                candidate,draws=generate_candidate(actual_parent,
                    PreparedFeedback(feedback['errors'],feedback['structured_feedback'],None,{},None),
                    broker,ordinal,plan['config']['runtime']['optimizer_structural_draws'],persist,
                    proposal_slot=slot,diagnosis_context=diagnosis_context)
            return dict(slot=slot,proposal_id=proposal_id,proposal_pattern_id=selection['selected_pattern_ids'][slot],
                diagnosis_source_prompt_sha256=parent.prompt_hash,parent_prompt=actual_parent.to_dict(),
                parent_prompt_sha256=actual_parent.prompt_hash,validation_before=score(actual_validation),
                optimize_before=score(actual_optimize),generation_draws=draws,
                generation_status='SKIPPED_NO_CURRENT_ERRORS' if not eligible else 'EXHAUSTED' if candidate is None
                    else 'NO_OP' if candidate==actual_parent else 'GENERATED',
                candidate_prompt=None if candidate is None else candidate.to_dict(),accepted=False,
                lineage=None if candidate is None else block_lineage(actual_parent,candidate),
                candidate_optimize=None,candidate_validation=None,candidate_optimize_score=None,candidate_validation_score=None,
                generation_cost=deepcopy(partial['generation_usage'].get(str(slot),{k:0 for k in before})) if saved_draws else delta(broker.usage_view(),before),evaluation_cost={k:0 for k in before},
                generation_seconds=0.0 if saved_draws else perf_counter()-clock,
                **({'generation_timing_status':'UNAVAILABLE_RESTORED_STAGE'} if saved_draws else {}),
                evaluation_seconds=0.0,repair_metrics=None,
                current_focus=view['focus'],same_block_as_first_accepted_edit=None)

        def evaluate(proposal,before_optimize):
            if proposal['generation_status']=='GENERATED':
                before=broker.usage_view();clock=perf_counter();candidate=SystemPrompt.from_dict(proposal['candidate_prompt'])
                co=_observations(optimize_rows,broker.evaluate(candidate,optimize_rows,'optimize',42,ordinal))
                cv=_observations(validation_rows,broker.evaluate(candidate,validation_rows,'validation',42,ordinal))
                proposal.update(generation_status='MEASURED',candidate_optimize=co,candidate_validation=cv,
                    candidate_optimize_score=score(co),candidate_validation_score=score(cv),
                    evaluation_cost={k:v+(partial['evaluation_usage'].get(str(proposal['slot']),{}).get(k,0) if restoring else 0) for k,v in delta(broker.usage_view(),before).items()},evaluation_seconds=perf_counter()-clock,
                    repair_metrics=repair_metrics(optimize,before_optimize,co,proposal['current_focus']['selected_support_ids']))

        if sequential:
            for slot,view in enumerate(views):
                proposal=generate(slot,view,current_parent,current_optimize,current_validation)
                evaluate(proposal,current_optimize)
                if proposal['generation_status']=='MEASURED':
                    proposal['accepted']=proposal['candidate_validation_score']>score(current_validation)
                    if slot and proposals[0]['accepted']:
                        proposal['same_block_as_first_accepted_edit']=(proposal['lineage']['edited_blocks']==proposals[0]['lineage']['edited_blocks'])
                    if proposal['accepted']:
                        current_parent=SystemPrompt.from_dict(proposal['candidate_prompt'])
                        current_optimize,current_validation=proposal['candidate_optimize'],proposal['candidate_validation']
                proposals.append(proposal)
        else:
            # Scientific parallelism: frozen same Parent, all generations before
            # any evaluation. Physical serial dispatch is explicit in the policy.
            proposals=[generate(slot,view,parent,optimize,validation) for slot,view in enumerate(views)]
            for proposal in proposals:evaluate(proposal,optimize)
            chosen=winner(proposals,score(validation))
            if chosen is not None:
                chosen['accepted']=True;current_parent=SystemPrompt.from_dict(chosen['candidate_prompt'])
                current_optimize,current_validation=chosen['candidate_optimize'],chosen['candidate_validation']
        for proposal in proposals:
            proposal['decision']='ACCEPTED_STRICT_VALIDATION_GAIN' if proposal['accepted'] else (
                ('REJECTED_PARALLEL_COMPETITION' if not sequential and proposal['candidate_validation_score']>proposal['validation_before']
                    else 'REJECTED_NO_STRICT_VALIDATION_GAIN') if proposal['generation_status']=='MEASURED' else proposal['generation_status'])
            persist(f"round_{ordinal}_proposal_{proposal['slot']}_result",proposal)
        accepted_ids=[p['proposal_id'] for p in proposals if p['accepted']]
        measured=any(p['generation_status']=='MEASURED' for p in proposals)
        decision='ACCEPTED_STRICT_VALIDATION_GAIN' if accepted_ids else 'REJECTED_NO_STRICT_VALIDATION_GAIN' if measured else (
            'SKIPPED_NO_OP' if any(p['generation_status']=='NO_OP' for p in proposals) else 'ABANDONED_GENERATION_RETRIES_EXHAUSTED')
        metric=dict(round=ordinal,accepted=bool(accepted_ids),decision=decision,validation_before=score(validation),
            optimize_before=score(optimize),winning_proposal_id=None if sequential or not accepted_ids else accepted_ids[0],
            accepted_proposal_ids=accepted_ids,accepted_update_count=len(accepted_ids),
            distinct_pattern_count=len({p['proposal_pattern_id'] for p in proposals if p['generation_draws']>0 and p['proposal_pattern_id'] is not None}),
            evaluated_distinct_pattern_count=len({p['proposal_pattern_id'] for p in proposals if p['generation_status']=='MEASURED' and p['proposal_pattern_id'] is not None}),
            candidate_evaluated=measured,proposals=proposals,shared_diagnostic_cost=diagnostic_cost,
            cost=delta(broker.usage_view(),start),wall_clock_seconds=perf_counter()-round_start,
            pair_changes=dict(status='SEQUENTIAL_DIFFERENT_PARENTS_NOT_SAME_PARENT_COMPARISON') if sequential else pair_changes(optimize,proposals,views),**selection)
        current=dict(scope_identity=scope_id,parent_prompt=parent.to_dict(),prompt=current_parent.to_dict(),
            optimize=current_optimize,validation=current_validation,module_state=None,metric=metric)
        _checkpoint(persist,f'checkpoint_{ordinal}',current)
        previous=dict(current,integrity_sha256=digest(current));metrics.append(metric);history.append(deepcopy(previous['prompt']))
    result=dict(cell=cell,scope_identity=scope_id,module_binding=binding,module_state=None,
        rounds_completed=len(metrics),prompt_history=history,metrics=metrics,final_prompt=previous['prompt'],
        final_optimize_correct=score(previous['optimize']),optimize_count=len(oi),
        final_validation_correct=score(previous['validation']),validation_count=len(vi),
        usage={k:broker.usage_view()[k]-v+old_usage[k] for k,v in begin.items()},
        search_wall_clock_seconds=bootstrap_seconds+sum(m['wall_clock_seconds'] for m in metrics))
    persist('result',result)
    return result
