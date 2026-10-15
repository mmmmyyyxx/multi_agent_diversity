"""Offline reconstruction of native sampling, retention, gates and final proposal."""
from copy import deepcopy
import json
from ...persistence.durable_io import read_json
from ...search.system_prompt import SystemPrompt,candidate_failed_checks
from ...search.schemas import SearchContractError
from ...benchmarks.data_freeze import digest
from .multi_parent import SinglePromptPool,score,parent_context,round_decision
from .engine import _observations
from ..single_prompt import batch_ids


def require(condition):
    if not condition:raise SearchContractError('MULTI_PARENT_AUDIT_MISMATCH')


def sealed(value):require(value.get('integrity_sha256')==digest({k:v for k,v in value.items() if k!='integrity_sha256'}))


def audit_cells(directory,scope,results=None,allow_partial=False):
    if allow_partial:raise SearchContractError('MULTI_PARENT_PARTIAL_REPLAY_NOT_BOUND')
    plan=scope['plan'];results=read_json(directory/'results.json') if results is None else results
    results=json.loads(json.dumps(results,allow_nan=False))
    require([r['cell'] for r in results]==plan['cells'])
    for result in results:
        cell=result['cell'];folder=directory/cell['cell_id'];n=plan['config']['rounds']
        cps=[read_json(folder/f'checkpoint_{i}.json') for i in range(n+1)]
        for cp in cps:sealed(cp)
        first=cps[0];ids=tuple(scope['optimize_sample_ids']);vids=tuple(scope['gate_membership']['sample_ids'])
        require(first['scope']['cell']==cell and first['scope']['plan_identity']==plan['plan_identity'])
        require(first['scope']['module_binding']==plan['module_bindings'][cell['arm_id']]==result['module_binding'])
        require(first['scope']['initial_prompt_sha256']==scope['initial_prompt_sha256'])
        require(SystemPrompt.from_dict(first['prompt']).prompt_hash==scope['initial_prompt_sha256'])
        require(first['scope']['optimize_membership_sha256']==digest(scope['optimize_sample_ids']))
        require(first['scope']['validation_membership_sha256']==digest(scope['gate_membership']['sample_ids']))
        sid=digest(first['scope']);require(first['scope_identity']==sid and result['scope_identity']==sid)
        pool=SinglePromptPool(ids);pool.update(SystemPrompt.from_dict(first['prompt']),first['optimize'])
        _observations([dict(stable_example_id=i) for i in vids],first['validation'])
        require(first['pool']==pool.export())
        validation_profiles={SystemPrompt.from_dict(first['prompt']).prompt_hash:first['validation']}
        require(first['validation_profiles']==validation_profiles)
        for ordinal in range(1,n+1):
            previous,cp=cps[ordinal-1],cps[ordinal];metric=cp['metric']
            require(cp['scope_identity']==sid and cp['parent_prompt']==previous['prompt'] and metric['round']==ordinal)
            require(not set(ids)&set(vids) and score(cp['validation'])>=score(previous['validation']))
            root=SystemPrompt.from_dict(previous['prompt']);parents,selection=pool.draw(cell['seed'],ordinal)
            _observations([dict(stable_example_id=i) for i in ids],cp['optimize'])
            _observations([dict(stable_example_id=i) for i in vids],cp['validation'])
            require(metric['selection']==selection and len(metric['slots'])==3)
            opportunity=read_json(folder/f'round_{ordinal}_opportunity.json')
            require(opportunity['selection']==selection and opportunity['parent_prompt']==previous['prompt'])
            errors=opportunity['errors'];require(len({e['example_id'] for e in errors})==len(errors))
            module=read_json(folder/f'round_{ordinal}_module.json')
            require(module['module_binding']==result['module_binding'] and module['module_state']==cp['module_state'])
            if cell['module']=='full_gradient_pattern':
                from .full_pattern import project_full
                audit=module['audit']
                projected,focus=project_full(audit['complete_diagnosis'],cell['seed'],ordinal,audit['sampling_context'])
                require(audit['sampling_context']['prompt_sha256']==root.prompt_hash and audit['focus']==focus)
                require(audit['sampling_context']['cell_id']==cell['cell_id'])
                require(projected['errors']==errors and projected['structured_feedback']==opportunity['structured_feedback']
                    and opportunity['module_context'] is None)
                wrong={i for i in ids if not previous['optimize'][i]['correct']}
                require({e['example_id'] for e in audit['complete_diagnosis']['errors']}==wrong)
                require({e['example_id'] for e in audit['complete_diagnosis']['structured_feedback']['diagnostics']}==wrong)
            else:
                wrong=tuple(i for i in ids if not previous['optimize'][i]['correct'])
                expected=wrong if cell['error_source']=='all_wrong' else batch_ids(wrong,cell['seed'],ordinal,3)
                require([e['example_id'] for e in errors]==[i for i in ids if i in set(expected)])
                require(opportunity['structured_feedback'] is None and opportunity['module_context'] is None)
            for e in errors:
                require(e['example_id'] in previous['optimize'] and previous['optimize'][e['example_id']]['correct'] is False)
            for slot,parent in enumerate(parents):
                s=metric['slots'][slot];require(s['slot']==slot and s['parent_prompt']==parent.prompt.to_dict())
                packet=read_json(folder/f'round_{ordinal}_slot_{slot}_opportunity.json')
                require(packet==dict(parent_prompt=s['parent_prompt'],errors=errors,
                    structured_feedback=opportunity['structured_feedback'],module_context=opportunity['module_context'],
                    parent_context=parent_context(root,parent,errors)))
                require(type(s['draws']) is int and 1<=s['draws']<=plan['config']['runtime']['optimizer_structural_draws'])
                generation=[read_json(folder/f'round_{ordinal}_slot_{slot}_generation_{d}.json') for d in range(1,s['draws']+1)]
                require(all(g['errors'] for g in generation[:-1]))
                g=generation[-1];require(g['candidate']==s['candidate'])
                if s['disposition']=='EXHAUSTED':
                    require(s['candidate'] is None and s['observations'] is None and g['errors']
                        and s['draws']==plan['config']['runtime']['optimizer_structural_draws'])
                else:
                    child=SystemPrompt.from_dict(s['candidate'])
                    require(not g['errors'] and not candidate_failed_checks(child,parent_prompt=parent.prompt))
                    if s['disposition']=='NO_OP':require(child==parent.prompt and s['observations'] is None)
                    else:
                        require(s['disposition']=='MEASURED' and child!=parent.prompt)
                        pool.update(child,_observations([dict(stable_example_id=i) for i in ids],s['observations']))
            measured=[s for s in metric['slots'] if s['disposition']=='MEASURED']
            winner=min(measured,key=lambda s:(-score(s['observations']),SystemPrompt.from_dict(s['candidate']).prompt_hash,s['slot'])) if measured else None
            cv=cp['candidate_validation'];require((cv is None)==(winner is None))
            if cv is not None:_observations([dict(stable_example_id=i) for i in vids],cv)
            if winner is not None:
                key=SystemPrompt.from_dict(winner['candidate']).prompt_hash
                require(key not in validation_profiles or validation_profiles[key]==cv)
                validation_profiles[key]=deepcopy(cv)
            require(cp['validation_profiles']==validation_profiles)
            accepted=cv is not None and score(cv)>score(previous['validation'])
            require(metric['accepted'] is accepted and metric['winner_slot']==(winner['slot'] if winner else None))
            require(metric['validation_before']==score(previous['validation']) and metric['validation_candidate']==(score(cv) if cv is not None else None))
            require(metric['decision']==round_decision(metric['slots'],winner,accepted))
            require(cp['pool']==pool.export())
            for key,new in [('prompt',winner['candidate'] if accepted else previous['prompt']),
                            ('optimize',winner['observations'] if accepted else previous['optimize']),
                            ('validation',cv if accepted else previous['validation'])]:require(cp[key]==new)
        final=read_json(folder/'final_selection.json');sealed(final);best=pool.best()
        require(final['proposal_prompt']==best.prompt.to_dict() and final['optimize']==best.observations)
        _observations([dict(stable_example_id=i) for i in vids],final['validation'])
        require(best.identity not in validation_profiles or validation_profiles[best.identity]==final['validation'])
        admitted=score(final['validation'])>=score(cps[-1]['validation'])
        require(final['admitted'] is admitted and final['running_prompt']==cps[-1]['prompt'] and final['running_validation']==cps[-1]['validation'])
        require(final['status']==('FINAL_PROPOSAL_ADMITTED' if admitted else 'FINAL_PROPOSAL_REJECTED'))
        require(result==read_json(folder/'result.json') and result['final_admission']=={k:v for k,v in final.items() if k!='integrity_sha256'})
        require(result['metrics']==[cp['metric'] for cp in cps[1:]] and result['prompt_history']==[cp['prompt'] for cp in cps])
        require(result['rounds_completed']==n and result['final_prompt']==best.prompt.to_dict())
        require(result['module_state']==cps[-1]['module_state'] and result['optimize_count']==len(ids) and result['validation_count']==len(vids))
        require(result['final_validation_correct']==score(final['validation']) and result['final_optimize_correct']==score(best.observations))
    return results
