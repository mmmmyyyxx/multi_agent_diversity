"""Read-only observations; never imported by search, admission or stopping.

Gradient shape annotations explain recorded failures. They do not validate or
repair a draw: a caller must independently replay the exact frozen validator.
"""
from collections import Counter
import json
import math


def opportunity_streaks(opportunities):
    """Measure deployed Vote after each complete opportunity, not candidate Vote.

    No-commit opportunities and no-Vote-improvement opportunities are distinct
    from the frozen eligible-member epochs used by the scientific stopper.
    """
    result=[]; no_commit=no_vote=0; previous=None
    for row in opportunities:
        committed=row['committed']; before=row['parent_vote']; after=row['child_vote']
        if type(committed) is not bool or any(type(x) not in (int,float) or not math.isfinite(x) for x in (before,after)):
            raise ValueError('INVALID_OPPORTUNITY_OBSERVATION')
        if previous is not None and before != previous:
            raise ValueError('DISCONTINUOUS_DEPLOYED_VOTE')
        if not committed and before != after:
            raise ValueError('VOTE_CHANGED_WITHOUT_COMMIT')
        no_commit=0 if committed else no_commit+1
        no_vote=0 if after>before else no_vote+1
        result.append(dict(ordinal=len(result)+1,committed=committed,
            deployed_vote_delta=after-before,consecutive_no_commit_opportunities=no_commit,
            consecutive_no_vote_improvement_opportunities=no_vote))
        previous=after
    return dict(rows=result,final_no_commit=no_commit,final_no_vote_improvement=no_vote,
        max_no_commit=max((r['consecutive_no_commit_opportunities'] for r in result),default=0),
        max_no_vote_improvement=max((r['consecutive_no_vote_improvement_opportunities'] for r in result),default=0),
        scientific_stop_inference='NONE_DIAGNOSTIC_OPPORTUNITIES_ARE_NOT_ELIGIBLE_MEMBER_EPOCHS')


def gradient_shape_annotation(text,finish_reason):
    """Closed field/length categories only, without any source-content judgment.

    Primary annotation follows the frozen V2.5 check order. All length breaches
    are also counted, so secondary issues cannot be mistaken for primary causes.
    """
    if finish_reason!='stop':return dict(primary='NONSTOP_PROVIDER_FINISH',issues=['NONSTOP_PROVIDER_FINISH'])
    try:value=json.loads(text)
    except (ValueError,TypeError) as error:
        category='JSON_SYNTAX_OR_ENVELOPE'
        if isinstance(text,str) and text.lstrip().startswith('```'):category='JSON_FENCED_ENVELOPE'
        elif isinstance(error,json.JSONDecodeError) and error.msg.startswith('Invalid \\escape'):category='JSON_INVALID_BACKSLASH_ESCAPE'
        elif isinstance(error,json.JSONDecodeError) and error.msg.startswith('Invalid control character'):category='JSON_UNESCAPED_CONTROL_CHARACTER'
        return dict(primary=category,issues=[category])
    fields={'disposition','observed_failure','diagnosis','reusable_correction','suggested_block','expected_effect'}
    if not isinstance(value,dict) or set(value)!=fields:
        return dict(primary='SCHEMA_KEYS_OR_OBJECT',issues=['SCHEMA_KEYS_OR_OBJECT'])
    issues=[]
    if value['disposition'] not in ('ACTIONABLE','UNCERTAIN'):issues.append('SCHEMA_DISPOSITION')
    if value['suggested_block'] not in (None,'role','strategy','answer'):
        block=value['suggested_block']
        issues.append('SCHEMA_BLOCK_CASE' if isinstance(block,str) and block.casefold() in ('role','strategy','answer') else 'SCHEMA_BLOCK')
    for field in ('observed_failure','diagnosis'):
        item=value[field]
        if not isinstance(item,str) or not item.strip():issues.append('SCHEMA_'+field.upper())
        elif len(item)>240:issues.append(field.upper()+'_OVER_240')
    if value['disposition']=='UNCERTAIN':
        if value['reusable_correction'] is not None or value['expected_effect'] is not None:
            issues.append('SCHEMA_UNCERTAIN_NULLS')
    elif value['disposition']=='ACTIONABLE':
        item=value['expected_effect']
        if not isinstance(item,str) or not item.strip():issues.append('SCHEMA_EXPECTED_EFFECT')
        elif len(item)>240:issues.append('EXPECTED_EFFECT_OVER_240')
        item=value['reusable_correction']
        if not isinstance(item,str) or not item.strip():issues.append('SCHEMA_REUSABLE_CORRECTION')
        elif len(item)>400:issues.append('REUSABLE_CORRECTION_OVER_400')
    return dict(primary=issues[0] if issues else 'REQUIRES_FROZEN_CONTENT_VALIDATION',issues=issues)


def gradient_cost_summary(groups):
    """Groups must already be joined and audited against frozen draw diagnostics.

    Retry costs are attributed to the preceding rejected draw, not to an inferred
    provider cause or the resulting retry's own category. All amounts are usage.
    """
    primary=Counter(); issue_counts=Counter(); reject_cost=Counter(); retry_trigger_cost=Counter()
    total=extra=inputs=outputs=0; first=accepted=exhausted=0; finish=Counter()
    exhausted_cost=recovered_extra=exhausted_extra=0
    for group in groups:
        if not 1<=len(group)<=3:raise ValueError('UNBOUND_GRADIENT_DRAW_COUNT')
        for i,draw in enumerate(group):
            if type(draw['accepted']) is not bool:raise ValueError('INVALID_DRAW_ACCEPTANCE')
            for key in ('input_tokens','output_tokens'):
                if type(draw[key]) is not int or draw[key]<0:raise ValueError('INVALID_DRAW_USAGE')
            if draw['accepted'] and i!=len(group)-1:raise ValueError('DRAW_AFTER_ACCEPTANCE')
            cost=draw['input_tokens']+draw['output_tokens']; total+=cost
            finish[draw['finish_reason']]+=1
            if not draw['accepted']:
                primary[draw['category']]+=1; issue_counts.update(draw['issues']); reject_cost[draw['category']]+=cost
            if i:
                extra+=cost; inputs+=draw['input_tokens']; outputs+=draw['output_tokens']
                retry_trigger_cost[group[i-1]['category']]+=cost
        if group[-1]['accepted']:
            accepted+=1; first+=len(group)==1
            recovered_extra+=sum(d['input_tokens']+d['output_tokens'] for d in group[1:])
        else:
            if len(group)!=3:raise ValueError('INCOMPLETE_EXHAUSTED_GRADIENT')
            exhausted+=1
            exhausted_cost+=sum(d['input_tokens']+d['output_tokens'] for d in group)
            exhausted_extra+=sum(d['input_tokens']+d['output_tokens'] for d in group[1:])
    return dict(logical_gradients=len(groups),physical_draws=sum(map(len,groups)),accepted=accepted,
        first_pass_accepted=first,recovered=accepted-first,exhausted=exhausted,
        extra_draws=sum(len(g)-1 for g in groups),total_usage_tokens=total,extra_usage_tokens=extra,
        extra_input_tokens=inputs,extra_output_tokens=outputs,primary_rejections=dict(primary),
        exhausted_total_usage_tokens=exhausted_cost,exhausted_extra_usage_tokens=exhausted_extra,
        successful_recovery_extra_usage_tokens=recovered_extra,
        overlapping_shape_issues=dict(issue_counts),rejected_usage_by_category=dict(reject_cost),
        retry_usage_by_preceding_rejection=dict(retry_trigger_cost),finish_reasons=dict(finish))
