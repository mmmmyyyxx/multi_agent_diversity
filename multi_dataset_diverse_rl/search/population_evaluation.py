"""Complete Optimize fitness, independent deployment guards and joint final team."""
from dataclasses import replace
from copy import deepcopy
from .binary_runtime import FixedPeerTeamEvaluationProvider, BinaryTeamStateStore
from .parent_population import Fitness, ParentPopulation, digest
from .schemas import SearchContractError, EvaluatedCandidate
from .system_prompt import team_identity,SystemPrompt

FINAL_POLICY=dict(identity='POPULATION_ARGMAX_JOINT_ADMISSION_V1',selection='one_closure_argmax_per_member',
    full='complete_optimize_joint_vote',initial_floor='all_changed_members',
    progress='not_required_at_final_selection_per_op_commit_progress_unchanged',
    shadow='joint_vote_nonregression_and_existing_member_loss_limit_each_changed_member',
    rejected='explicit_rejected_proposal_no_silent_fallback',reselection=False)

class PopulationEvaluationProvider(FixedPeerTeamEvaluationProvider):
    def __init__(self,store,transition,population,*,fitness_policy=None):
        from .measurement_policies import CompleteOptimizeFitness
        self.fitness_policy=fitness_policy or CompleteOptimizeFitness()
        super().__init__(store,transition);self.population=population;self.member_profiles={};self.known_prompts={}
        self.initial_prompts=store.prompts
    def bootstrap(self):
        if any(self.population.members.values()):return
        state=self.store.snapshot()
        for m,p in enumerate(state.member_prompts):
            self.known_prompts[p.prompt_hash]=p
            self.member_profiles[(m,p.prompt_hash)]=state.diagnostics['raw_profiles'][m]
            accuracy=self.fitness(state,m,'MEMBER_ACCURACY_PROXY','initial:'+str(m))
            self.population.update(m,p,self.fitness(state,m,'TEAM_VOTE','initial:'+str(m)),accuracy)
        self.population.save()
    def fitness(self,state,member,source,observation_id):
        return self.fitness_policy.assign(state,member,source,observation_id,len(self.store.examples))
    def profile(self,member,prompt):
        self.known_prompts[prompt.prompt_hash]=prompt
        key=(member,prompt.prompt_hash)
        if key not in self.member_profiles:
            solver=self.store.solver
            if hasattr(solver,'observe_member'):solver.observe_member(member)
            items=[e.item for e in self.store.examples]
            batch=getattr(solver,'solve_batch',None)
            value=tuple(batch(prompt,items,stage='full',split='optimize') if callable(batch) else
                (solver.solve(prompt,e,stage='full',split='optimize') for e in items))
            if len(value)!=len(items):raise SearchContractError('COMPLETE_MEMBER_PROFILE_REQUIRED')
            self.member_profiles[key]=value
        return self.member_profiles[key]
    def joint_snapshot(self,prompts):
        if len(prompts)!=5:raise SearchContractError('FINAL_TEAM_SHAPE_INVALID')
        scratch=BinaryTeamStateStore(benchmark=self.store.benchmark,examples=self.store.examples,
            prompts=prompts,solver=self.store.solver,aggregation=self.store.aggregation,freeze_initial_competence=True)
        scratch.profiles={m:self.profile(m,p) for m,p in enumerate(prompts)}
        return scratch.snapshot()
    def _evaluate(self,opportunity,candidate,ids,stage):
        if stage!='full' or ids!={e.item.input_id for e in self.store.examples}:
            raise SearchContractError('POPULATION_REQUIRES_COMPLETE_OPTIMIZE')
        parent=self._parent(opportunity)
        if candidate.backend_details.get('parent_state_id')!=parent.team_state_id:
            raise SearchContractError('CANDIDATE_PARENT_MISMATCH')
        t=opportunity.target_member
        prompts=tuple(candidate.prompt if m==t else p for m,p in enumerate(parent.member_prompts))
        child=self.joint_snapshot(prompts)
        self.store.full_profiles[(parent.team_state_id,t,candidate.prompt)]=self.profile(t,candidate.prompt)
        self._last_repeated_ids={e.item.input_id for e,p in zip(self.store.examples,
            self.profile(t,candidate.prompt),strict=True) if isinstance(p,dict) and
            p.get('solver_trajectory',{}).get('retry_summary',{}).get('repeated_format_failure') is True}
        return parent.diagnostics['team_states'],child.diagnostics['team_states']
    def parent_observations(self,member,prompt,rows):
        from ..benchmarks.math_response_evidence import adaptive_trajectory
        if (member,prompt.prompt_hash) not in self.member_profiles:
            raise SearchContractError('PARENT_OBSERVATION_NOT_PREVIOUSLY_MEASURED')
        indexed={e.item.input_id:(e,p) for e,p in zip(self.store.examples,self.member_profiles[(member,prompt.prompt_hash)],strict=True)}
        result={}
        for row in rows:
            e,p=indexed[row.example_id];parsed=self.store.benchmark.parse_member_output(p,e.item)
            result[row.example_id]=dict(prompt_sha256=prompt.prompt_hash,source='actual_cached_complete_optimize_response',
                correct=self.store.benchmark.score_member_output(parsed,e.reference)==1,
                trajectory=adaptive_trajectory(p,member_id=member,prompt=prompt,example_id=e.item.input_id,
                    problem=self.store.benchmark.format_input(e.item)))
        return result

class PopulationEvaluationPipeline:
    def __init__(self,provider,*,pool_update=None):
        from .measurement_policies import MeasuredPoolUpdate
        self.provider=provider;self.pool_update=pool_update or MeasuredPoolUpdate()
    async def evaluate(self,opportunity,search):
        if len(search.candidates)>3:raise SearchContractError('POPULATION_SLOT_CEILING')
        rows=[];seen=set();seen_slots=set()
        plan=search.search_state.get('selection_plan')
        if search.candidates and (not isinstance(plan,dict) or plan.get('opportunity_id')!=opportunity.opportunity_id
                or plan.get('member')!=opportunity.target_member
                or self.provider.population.plans.get(digest([plan['member'],plan['seed'],plan['opportunity_id']]))!=plan):
            raise SearchContractError('CANDIDATE_SELECTION_PLAN_NOT_FROZEN')
        for c in search.candidates:
            d=c.backend_details
            if (d.get('opportunity_id')!=opportunity.opportunity_id or not d.get('contract_valid')
                    or d.get('parent_state_id')!=opportunity.parent_state_id):
                raise SearchContractError('POPULATION_CANDIDATE_INVALID')
            slot=d.get('slot')
            if (type(slot) is not int or slot not in range(3) or slot in seen_slots
                    or d.get('selected_parent_sha256')!=plan['selected_prompt_sha256'][slot]
                    or SystemPrompt.from_dict(d['selected_parent']).prompt_hash!=d['selected_parent_sha256']):
                raise SearchContractError('CANDIDATE_SELECTED_PARENT_MISMATCH')
            seen_slots.add(slot)
            seen.add(c.prompt.prompt_hash)
            selected_parent=SystemPrompt.from_dict(d['selected_parent'])
            from .system_prompt import candidate_failed_checks
            if candidate_failed_checks(c.prompt,parent_prompt=selected_parent):
                raise SearchContractError('POPULATION_EDIT_SCOPE_INVALID')
            if c.prompt==selected_parent:
                # The generator normally records this before evaluation. Keep
                # this defensive boundary free of no-op metric calls as well.
                if isinstance(search.search_state,dict):
                    search.search_state.setdefault('evaluation_skips',[]).append(dict(candidate_id=c.candidate_id,disposition='NO_OP'))
                continue
            if not d.get('changed'):raise SearchContractError('POPULATION_CHANGE_METADATA_MISMATCH')
            full=await self.provider.full(opportunity,c)
            t=opportunity.target_member
            prompts=list(self.provider.store.snapshot().member_prompts);prompts[t]=c.prompt
            state=self.provider.joint_snapshot(tuple(prompts))
            proxy=self.provider.fitness(state,t,'MEMBER_ACCURACY_PROXY',c.candidate_id)
            vote=self.provider.fitness(state,t,'TEAM_VOTE',c.candidate_id)
            from .optimization_evidence import coverage_effect
            selected_parent=SystemPrompt.from_dict(d['selected_parent'])
            def own(prompt):
                profile=self.provider.profile(t,prompt)
                return {e.item.input_id:dict(correct=self.provider.store.benchmark.score_member_output(
                    self.provider.store.benchmark.parse_member_output(p,e.item),e.reference)==1,
                    valid=self.provider.store.benchmark.parse_member_output(p,e.item).valid)
                    for e,p in zip(self.provider.store.examples,profile,strict=True)}
            effect=coverage_effect(own(selected_parent),own(c.prompt),scope='complete_optimize_actual_edit_parent')
            c=replace(c,backend_details={**d,'selected_parent_member_effect':dict(
                fixed=len(effect['fixed_ids']),broken=len(effect['broken_ids']),count=len(self.provider.store.examples)),
                'solver_evaluated':True})
            update=self.pool_update.update(self.provider.population,t,c.prompt,vote,proxy)
            self.provider.population.save()
            full=replace(full,aggregation_diagnostics={**full.aggregation_diagnostics,
                'fitness_provenance':dict(member_accuracy_proxy=proxy.__dict__,actual_vote=vote.__dict__),
                'pool_update':update,'evaluation_policy':'ALL_LEGAL_THREE_COMPLETE_OPTIMIZE'})
            rows.append(EvaluatedCandidate(c,None,full,True,False,
                dict(target_member=t,population_update=update,**{k:v for k,v in full.aggregation_diagnostics.items()
                    if k in {'promotion_reason','scientific_risk_code','operational_failure'}})))
        return tuple(rows)

class PopulationFinalizer:
    def __init__(self,provider,gate,transition,*,admission=None):
        from .measurement_policies import JointFinalAdmission
        self.admission=admission or JointFinalAdmission()
        self.provider=provider;self.gate=gate;self.transition=transition
    async def finalize(self):
        store=self.provider.store;pool=self.provider.population
        proposed,selection=pool.best_team();baseline=store.snapshot()
        self.proposed_prompts=proposed
        self.pending=dict(selection=selection,proposed_team_state_id=team_identity(proposed),
            status='FINAL_ASSEMBLY_NOT_EVALUABLE',admitted=False)
        # Selection is frozen before measuring the new joint team. Never select
        # again using its outcomes, including when the assembly is rejected.
        child=self.provider.joint_snapshot(proposed)
        self.proposed_state=child
        changed=[m for m in range(5) if proposed[m]!=baseline.member_prompts[m]]
        invalid=sum(not all(s.team_validity) for s in child.diagnostics['team_states'])
        parent_invalid=sum(not all(s.team_validity) for s in baseline.diagnostics['team_states'])
        allowed=self.admission.allowed(child=child,baseline=baseline,changed=changed,
            initial_scores=self.transition.initial_scores,invalid=invalid,parent_invalid=parent_invalid,
            invalidity=self.provider.invalid_predictions_are_incorrect)
        shadow=await self.gate.check_team(proposed,baseline) if allowed else None
        accepted=bool(allowed and shadow)
        for m,p in enumerate(proposed):
            old=pool.members[m][p.prompt_hash]
            pool.update(m,p,self.provider.fitness(child,m,'TEAM_VOTE','final_joint'),old.member_accuracy)
        pool.save()
        if accepted:store.restore(child)
        result=dict(identity=FINAL_POLICY['identity'],selection=selection,
            proposed_team_state_id=child.team_state_id,proposed_prompt_sha256=[p.prompt_hash for p in proposed],
            baseline_team_state_id=baseline.team_state_id,joint_vote=child.team_scores['vote_correct_count'],
            joint_member_scores=child.member_scores,count=len(store.examples),changed_members=changed,
            full_passed=allowed,shadow_passed=shadow,admitted=accepted,
            status='FINAL_ASSEMBLY_ADMITTED' if accepted else 'FINAL_ASSEMBLY_REJECTED',
            deployed_team_state_id=store.snapshot().team_state_id,selection_repeated=False)
        self.pending=result
        return result
