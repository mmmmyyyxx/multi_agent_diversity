"""One meaningful candidate per opportunity, with a strict Validation100 gate."""
import json
from ..local_optimizers.schemas import LocalEvidenceExample
from ..search.system_prompt import SystemPrompt,candidate_failed_checks,block_lineage
from ..search.schemas import SearchContractError
from ..search.optimizer_output_contract import parse_output
from ..search.generation_failures import GeneratedOutputFailure
from .single_prompt import OPTIMIZER_PROMPT,raw_error,batch_ids,structured_feedback
from .single_prompt_runtime import StudyBroker
from .gated_prompt_policy import policy

GENERATOR_PROMPT=OPTIMIZER_PROMPT+'''
The candidate must actually change the Prompt. Do not return an unchanged block.
The supplied block_character_limits are exact COMPLETE replacement budgets.
Rewrite, shorten or reorganize the block when space is tight; do not keep
appending checklists. Stay within the selected block's character limit.
'''


class GatedPromptBroker(StudyBroker):
    def study_policy(self):return policy()
    def timeout_policy(self):return self.study_policy()['timeout_recovery']
    def legal_role_split(self,role,split):
        return self.phase=='train' and ((split=='optimize' and role in ('solver','reflection','pattern_gradient','pattern_cluster'))
            or (split=='validation' and role=='solver'))
    def cache_ordinal(self,split):return 0 if split=='validation' else self.round_index


def replacement_budgets(prompt):
    return {k:3000-4-sum(len(v) for b,v in prompt.to_dict().items() if b!=k) for k in ('role','strategy','answer')}


def generate_changed_candidate(prompt,feedback,broker,ordinal,persist_attempt):
    packet=dict(current_system_prompt=prompt.to_dict(),errors=feedback['errors'],
        structured_feedback=feedback['structured_feedback'],
        block_character_limits=replacement_budgets(prompt),
        full_prompt_characters=len(prompt.render()),maximum_prompt_characters=3000,
        objective='Improve future accuracy with one genuinely changed legal block')
    examples=tuple(LocalEvidenceExample(x['example_id'],x['problem'],x['reference']) for x in feedback['errors'])
    for draw in range(1,policy()['optimizer_structural_draws']+1):
        response=broker.complete(role='reflection',split='optimize',stage=f'round_{ordinal}_legal_candidate_draw_{draw}',
            messages=[dict(role='system',content=GENERATOR_PROMPT),dict(role='user',content=json.dumps(packet,sort_keys=True,ensure_ascii=True))])
        errors=[]; child=None; proposed_length=None
        try:value=parse_output(response['text'],stage='reflection')
        except GeneratedOutputFailure as exc:errors=[str(exc)]
        else:
            if response.get('finish_reason')!='stop':errors=['output_truncated']
            elif not isinstance(value,dict) or set(value)!={'block','content'}:errors=['invalid_output_schema']
            elif value['block'] not in ('role','strategy','answer'):errors=['invalid_block']
            else:
                try:child=prompt.edit(value['block'],value['content'])
                except SearchContractError as exc:
                    if str(exc) not in ('SYSTEM_PROMPT_BLOCK_INVALID','SYSTEM_PROMPT_OVER_LENGTH'):raise
                    errors=[str(exc)]
                    if isinstance(value['content'],str):
                        proposed_length=4+sum(len(value['content']) if k==value['block'] else len(v) for k,v in prompt.to_dict().items())
                if not errors:errors=list(candidate_failed_checks(child,parent_prompt=prompt,examples=examples))
        audit=dict(draw=draw,errors=errors,request_sha256=response['request_sha256'],
            proposed_prompt_characters=proposed_length,block_character_limits=packet['block_character_limits'])
        persist_attempt(f'round_{ordinal}_generation_{draw}',dict(audit=audit,parent_prompt=prompt.to_dict(),
            candidate=child.to_dict() if child is not None and not errors else None))
        if not errors:return child,draw
        packet['output_contract_feedback']=dict(stage='reflection',errors=errors,
            proposed_prompt_characters=proposed_length,maximum_prompt_characters=3000,
            instruction='Return a changed complete replacement; shorten or rewrite to fit the exact block budget.')
    return None,draw


def run_cell(cell,optimize_rows,validation_rows,broker,persist):
    p=policy(); prompt=SystemPrompt.from_dict(cell['initial_prompt']); start_usage=broker.usage_view()
    optimize=broker.evaluate(prompt,optimize_rows,'optimize',cell['seed'],0)
    validation=broker.evaluate(prompt,validation_rows,'validation',cell['seed'],0)
    score=lambda rows:sum(v['correct'] for v in rows.values())
    persist('checkpoint_0',dict(prompt=prompt.to_dict(),optimize=optimize,validation=validation))
    metrics=[]; history=[prompt.to_dict()]
    for ordinal in range(1,p['rounds']+1):
        broker.round_index=ordinal
        before_usage=broker.usage_view()
        wrong=tuple(r['stable_example_id'] for r in optimize_rows if not optimize[r['stable_example_id']]['correct'])
        selected=wrong if cell['arm']=='full_errors' else batch_ids(wrong,cell['seed'],ordinal,p['error_batch_size'])
        rows=tuple(r for r in optimize_rows if r['stable_example_id'] in set(selected))
        structure=structured_feedback(prompt,rows,optimize,broker) if cell['arm']=='gradient_pattern' else None
        feedback=dict(errors=[raw_error(r,optimize[r['stable_example_id']]) for r in rows],structured_feedback=structure)
        persist(f'round_{ordinal}_opportunity',dict(parent_prompt=prompt.to_dict(),feedback=feedback,
            retained_validation_correct=score(validation),round_complete=False))
        candidate,draws=generate_changed_candidate(prompt,feedback,broker,ordinal,persist)
        if candidate is None:
            candidate_optimize=candidate_validation=None
            candidate_score=None;accepted=False;decision='ABANDONED_GENERATION_RETRIES_EXHAUSTED'
        else:
            candidate_optimize=broker.evaluate(candidate,optimize_rows,'optimize',cell['seed'],ordinal)
            candidate_validation=broker.evaluate(candidate,validation_rows,'validation',cell['seed'],ordinal)
            candidate_score=score(candidate_validation);accepted=candidate_score>score(validation)
            decision='ACCEPTED_STRICT_VALIDATION_GAIN' if accepted else 'REJECTED_NO_STRICT_VALIDATION_GAIN'
        incumbent_score=score(validation)
        metric=dict(round=ordinal,accepted=accepted,decision=decision,candidate_evaluated=candidate is not None,
            validation_before=incumbent_score,validation_candidate=candidate_score,
            optimize_before=score(optimize),optimize_candidate=score(candidate_optimize) if candidate_optimize is not None else None,
            optimizer_example_count=len(rows),generation_draws=draws,
            cost={k:broker.usage_view()[k]-v for k,v in before_usage.items()},
            lineage=block_lineage(prompt,candidate) if candidate is not None else None)
        # Preserve the incumbent observations as well as its Prompt on rejection.
        parent=prompt
        if accepted:prompt,optimize,validation=candidate,candidate_optimize,candidate_validation
        persist(f'checkpoint_{ordinal}',dict(prompt=prompt.to_dict(),optimize=optimize,validation=validation,
            candidate_prompt=candidate.to_dict() if candidate is not None else None,candidate_optimize=candidate_optimize,candidate_validation=candidate_validation,
            parent_prompt=parent.to_dict(),metric=metric,optimizer_feedback=feedback))
        metrics.append(metric); history.append(prompt.to_dict())
    result=dict(cell_id=cell['cell_id'],arm=cell['arm'],seed=cell['seed'],rounds_completed=len(metrics),
        opportunities_abandoned=sum(m['decision']=='ABANDONED_GENERATION_RETRIES_EXHAUSTED' for m in metrics),
        candidates_evaluated=sum(m['candidate_evaluated'] for m in metrics),
        prompt_history=history,metrics=metrics,final_prompt=prompt.to_dict(),
        final_optimize_correct=score(optimize),final_validation_correct=score(validation),
        final_selection=p['final_selection'],usage={k:broker.usage_view()[k]-v for k,v in start_usage.items()})
    persist('result',result)
    return result
