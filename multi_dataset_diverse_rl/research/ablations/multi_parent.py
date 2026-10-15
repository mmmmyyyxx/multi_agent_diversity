"""One measured Prompt pool, using the production population strategies directly.

No five-member container, dummy peers, Vote or Group evidence is constructed.
Pool retention and deployment are deliberately different decisions.
"""
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from ...benchmarks.data_freeze import digest
from ...benchmarks.data_freeze import file_hash
from ...search.system_prompt import SystemPrompt,block_lineage,candidate_failed_checks
from ...search.schemas import SearchContractError
from ...search.population_policies import FitnessWeightedSampling,BestThreeRetention,PerMemberArgmaxAssembly
from ..single_prompt import batch_ids,raw_error
from .interfaces import OptimizeContext,OptimizeOutcome,PreparedFeedback,ModuleProvider

POLICY=dict(identity='SINGLE_PROMPT_NATIVE_PARENT_POLICIES_V1',capacity=3,slots=3,
    sampling=FitnessWeightedSampling.identity,retention=BestThreeRetention.identity,
    final_selection=PerMemberArgmaxAssembly.identity,fitness='complete_optimize_accuracy',
    pool_admission='all_measured_children_independent_of_validation',
    winner='highest_optimize_then_prompt_sha256_then_slot',
    validation='one_winner_strict_gain_vs_running_root',final_gate='nonregression_vs_running_root',
    diagnosis='once_per_round_running_root_shared_across_slots',
    sampling_scope='scientific_policy_membership_no_repeat_or_arm_salt',
    no_op='skip_slot_no_regeneration',format_draws_per_slot=4,
    final_rejection='explicit_proposal_rejected_no_silent_incumbent_substitution',
    checkpoint_resume='whole_boundary_only_fresh_resume_contract_required')
POLICY['production_source_sha256']=file_hash(Path(__file__).resolve().parents[2]/'search/population_policies.py')
POLICY['provider_generation_seed']='not_exposed_by_current_generation_contracts'


def score(observations):return sum(v['correct'] for v in observations.values())


@dataclass(frozen=True)
class Accuracy:
    correct:int
    count:int
    @property
    def value(self):return self.correct/self.count


@dataclass(frozen=True)
class MeasuredParent:
    prompt:SystemPrompt
    observations:dict
    @property
    def identity(self):return self.prompt.prompt_hash
    @property
    def fitness(self):return Accuracy(score(self.observations),len(self.observations))


class SinglePromptPool:
    def __init__(self,ids):
        self.ids=tuple(ids);self.parents={}
        if not self.ids or len(set(self.ids))!=len(self.ids):raise SearchContractError('PARENT_MEMBERSHIP_REQUIRED')
        self.sampling=FitnessWeightedSampling();self.retention=BestThreeRetention();self.assembly=PerMemberArgmaxAssembly()
        self.sampling_scope=dict(identity=POLICY['identity'],support_sha256=digest(self.ids),support_count=len(self.ids))

    def ordered(self):return tuple(sorted(self.parents.values(),key=lambda p:(-p.fitness.value,p.identity)))

    def update(self,prompt,observations):
        from .engine import _observations
        observations=_observations([dict(stable_example_id=i) for i in self.ids],observations)
        old=self.parents.get(prompt.prompt_hash)
        if old is not None and old.observations!=observations:
            raise SearchContractError('PARENT_SAME_PROMPT_OBSERVATION_CHANGED')
        values={**self.parents,prompt.prompt_hash:MeasuredParent(prompt,observations)}
        ordered=tuple(sorted(values.values(),key=lambda p:(-p.fitness.value,p.identity)))
        kept=self.retention.keep(ordered)
        self.parents={p.identity:p for p in kept}

    def draw(self,seed,ordinal):
        parents=self.ordered();probabilities=self.sampling.probabilities(parents)
        chosen=self.sampling.draw(parents,probabilities,scope=self.sampling_scope,
            seed=seed,opportunity_id=f'round_{ordinal}',member=0)
        return chosen,dict(round=ordinal,seed=seed,scope=deepcopy(self.sampling_scope),
            distribution=[dict(prompt_sha256=p.identity,probability=q,correct=p.fitness.correct,count=p.fitness.count)
                for p,q in zip(parents,probabilities,strict=True)],selected_prompt_sha256=[p.identity for p in chosen])

    def best(self):return self.assembly.select((self.ordered(),))[0]

    def export(self):return [dict(prompt=p.prompt.to_dict(),observations=deepcopy(p.observations),
        prompt_sha256=p.identity,correct=p.fitness.correct,count=p.fitness.count) for p in self.ordered()]


def parent_context(root,parent,errors):
    """Root diagnoses remain labeled; Parent responses are actual measured results."""
    from ...benchmarks.math_response_evidence import adaptive_trajectory
    from ...search.group_diagnosis import readable_observation
    rows=[]
    for error in errors:
        ident=error['example_id'];observed=parent.observations[ident]
        trajectory=adaptive_trajectory(observed['profile'],member_id=0,prompt=parent.prompt,
            example_id=ident,problem=error['problem'])
        rows.append(dict(example_id=ident,prompt_sha256=parent.identity,
            **readable_observation(trajectory,correct=observed['correct'])))
    return dict(root_prompt_sha256=root.prompt_hash,selected_parent_sha256=parent.identity,
        diagnosis_source='running_root_not_selected_parent',selected_parent_observations=rows)


class SlotBroker:
    def __init__(self,broker,slot):self.broker,self.slot=broker,slot
    def complete(self,**kwargs):
        kwargs['stage']=kwargs['stage'].replace('_legal_candidate_',f'_slot_{self.slot}_legal_candidate_')
        return self.broker.complete(**kwargs)


def round_decision(slots,winner,accepted):
    if winner is not None:return 'ACCEPTED_STRICT_VALIDATION_GAIN' if accepted else 'REJECTED_NO_STRICT_VALIDATION_GAIN'
    return 'SKIPPED_NO_OP' if any(s['disposition']=='NO_OP' for s in slots) else 'ABANDONED_GENERATION_RETRIES_EXHAUSTED'


def run_cell(plan,cell,initial_prompt,optimize_rows,validation_rows,broker,persist,*,registry=None,resume=None):
    from .config import validate_plan
    from .modules import default_registry
    from .engine import _observations,_state,_checkpoint
    from .generation import generate_candidate
    if resume is not None:raise SearchContractError('MULTI_PARENT_FRESH_RESUME_CONTRACT_REQUIRED')
    registry=registry or default_registry();validate_plan(plan,registry)
    if cell not in plan['cells'] or broker.phase!='train':raise SearchContractError('MULTI_PARENT_CELL_PHASE')
    optimize_rows,validation_rows=tuple(optimize_rows),tuple(validation_rows)
    oi=tuple(r['stable_example_id'] for r in optimize_rows);vi=tuple(r['stable_example_id'] for r in validation_rows)
    if not vi or len(set(vi))!=len(vi) or set(oi)&set(vi):raise SearchContractError('MULTI_PARENT_SPLIT_MEMBERSHIP')
    broker.bind_cell(cell,plan);p=plan['config']['runtime'];root=SystemPrompt.from_dict(initial_prompt)
    binding=plan['module_bindings'][cell['arm_id']];module=registry.create(cell['module'])
    state=_state(module.initial_state(),binding);begin=broker.usage_view()
    optimize=_observations(optimize_rows,broker.evaluate(root,optimize_rows,'optimize',cell['seed'],0))
    validation=_observations(validation_rows,broker.evaluate(root,validation_rows,'validation',cell['seed'],0))
    validation_profiles={root.prompt_hash:deepcopy(validation)}
    pool=SinglePromptPool(oi);pool.update(root,optimize)
    scope=dict(plan_identity=plan['plan_identity'],cell=cell,initial_prompt_sha256=root.prompt_hash,
        optimize_membership_sha256=digest(oi),validation_membership_sha256=digest(vi),module_binding=binding)
    scope_id=digest(scope);history=[root.to_dict()];metrics=[]
    _checkpoint(persist,'checkpoint_0',dict(scope_identity=scope_id,scope=scope,prompt=root.to_dict(),
        optimize=optimize,validation=validation,pool=pool.export(),module_state=state,validation_profiles=deepcopy(validation_profiles)))
    for ordinal in range(1,p['rounds']+1):
        broker.round_index=ordinal;before=broker.usage_view();old=root
        parents,selection=pool.draw(cell['seed'],ordinal)  # All frozen before diagnosis/generation/upserts.
        wrong=tuple(i for i in oi if not optimize[i]['correct'])
        selected=wrong if cell['error_source']=='all_wrong' else batch_ids(wrong,cell['seed'],ordinal,3)
        selected_rows=tuple(r for r in optimize_rows if r['stable_example_id'] in set(selected))
        context=OptimizeContext(cell['cell_id'],cell['seed'],cell['repeat_id'],ordinal,root,
            selected_rows,{i:deepcopy(optimize[i]) for i in selected})
        provider=ModuleProvider(broker,binding['semantic_calls_per_opportunity'])
        prepared=module.prepare(context,deepcopy(state),provider)
        if not isinstance(prepared,PreparedFeedback):raise SearchContractError('MULTI_PARENT_FEEDBACK_TYPE')
        originals={r['stable_example_id']:raw_error(r,optimize[r['stable_example_id']]) for r in selected_rows}
        supplied=[e['example_id'] for e in prepared.errors]
        if len(set(supplied))!=len(supplied) or any(e!=originals.get(e['example_id']) for e in prepared.errors):
            raise SearchContractError('MULTI_PARENT_FEEDBACK_PROVENANCE')
        state=_state(prepared.state,binding)
        persist(f'round_{ordinal}_module',dict(module_binding=binding,audit=prepared.audit,module_state=state))
        persist(f'round_{ordinal}_opportunity',dict(parent_prompt=root.to_dict(),errors=prepared.errors,
            structured_feedback=prepared.structured_feedback,module_context=prepared.module_context,selection=selection))
        slots=[]
        for slot,parent in enumerate(parents):
            context_view=parent_context(root,parent,prepared.errors)
            opportunity=dict(parent_prompt=parent.prompt.to_dict(),errors=prepared.errors,
                structured_feedback=prepared.structured_feedback,module_context=prepared.module_context,parent_context=context_view)
            persist(f'round_{ordinal}_slot_{slot}_opportunity',opportunity)
            def save(name,value):persist(name.replace('_generation_',f'_slot_{slot}_generation_'),value)
            child,draws=generate_candidate(parent.prompt,prepared,SlotBroker(broker,slot),ordinal,
                p['optimizer_structural_draws'],save,parent_context=context_view)
            observation=None;disposition='EXHAUSTED' if child is None else 'NO_OP' if child==parent.prompt else 'MEASURED'
            if disposition=='MEASURED':
                observation=_observations(optimize_rows,broker.evaluate(child,optimize_rows,'optimize',cell['seed'],ordinal))
                pool.update(child,observation)
            slots.append(dict(slot=slot,parent_prompt=parent.prompt.to_dict(),candidate=child.to_dict() if child else None,
                observations=observation,draws=draws,disposition=disposition))
        measured=[s for s in slots if s['disposition']=='MEASURED']
        winner=min(measured,key=lambda s:(-score(s['observations']),SystemPrompt.from_dict(s['candidate']).prompt_hash,s['slot'])) if measured else None
        candidate_validation=None;accepted=False;candidate=None
        if winner is not None:
            candidate=SystemPrompt.from_dict(winner['candidate'])
            if candidate.prompt_hash not in validation_profiles:
                validation_profiles[candidate.prompt_hash]=_observations(validation_rows,broker.evaluate(candidate,validation_rows,'validation',cell['seed'],ordinal))
            candidate_validation=deepcopy(validation_profiles[candidate.prompt_hash])
            accepted=score(candidate_validation)>score(validation)
        decision=round_decision(slots,winner,accepted)
        # No Validation signals are delivered to modules, Parent sampling or pool retention.
        for s in slots:
            child=SystemPrompt.from_dict(s['candidate']) if s['candidate'] is not None else None
            state=_state(module.observe(OptimizeOutcome(ordinal,SystemPrompt.from_dict(s['parent_prompt']),child,
                candidate if accepted else root,deepcopy(next(x.observations for x in parents if x.prompt.to_dict()==s['parent_prompt'])),
                deepcopy(s['observations']),s['disposition']),deepcopy(state)),binding)
        metric=dict(round=ordinal,decision=decision,accepted=accepted,validation_before=score(validation),
            validation_candidate=score(candidate_validation) if candidate_validation is not None else None,
            winner_slot=winner['slot'] if winner else None,selection=selection,slots=slots,
            cost={k:broker.usage_view()[k]-v for k,v in before.items()})
        if accepted:root,optimize,validation=candidate,winner['observations'],candidate_validation
        _checkpoint(persist,f'checkpoint_{ordinal}',dict(scope_identity=scope_id,prompt=root.to_dict(),
            parent_prompt=old.to_dict(),optimize=optimize,validation=validation,pool=pool.export(),
            module_state=state,candidate_validation=candidate_validation,metric=metric,validation_profiles=deepcopy(validation_profiles)))
        metrics.append(metric);history.append(root.to_dict())
    proposal=pool.best()
    if proposal.identity not in validation_profiles:
        validation_profiles[proposal.identity]=_observations(validation_rows,broker.evaluate(proposal.prompt,validation_rows,'validation',cell['seed'],p['rounds']+1))
    final_validation=deepcopy(validation_profiles[proposal.identity])
    admitted=score(final_validation)>=score(validation)
    final=dict(proposal_prompt=proposal.prompt.to_dict(),optimize=proposal.observations,validation=final_validation,
        running_prompt=root.to_dict(),running_validation=validation,admitted=admitted,
        status='FINAL_PROPOSAL_ADMITTED' if admitted else 'FINAL_PROPOSAL_REJECTED')
    _checkpoint(persist,'final_selection',final)
    result=dict(cell=cell,scope_identity=scope_id,rounds_completed=len(metrics),module_binding=binding,
        final_prompt=proposal.prompt.to_dict(),final_admission=final,prompt_history=history,metrics=metrics,
        final_optimize_correct=score(proposal.observations),optimize_count=len(oi),
        final_validation_correct=score(final_validation),validation_count=len(vi),module_state=state,
        usage={k:broker.usage_view()[k]-v for k,v in begin.items()})
    persist('result',result)
    return result
