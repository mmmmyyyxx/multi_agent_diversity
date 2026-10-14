"""Atomic opportunity-boundary recovery, never paid-run authorization."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from ..persistence.durable_io import atomic_write_json,read_json
from .parent_population import digest,ParentPopulation
from .population_memory import PopulationEditMemory
from .schemas import SearchContractError,TransitionRecord
from .system_prompt import SystemPrompt,team_identity
from .history import HistoryState

SCHEMA='V211_GROUP_COMPLETE_SEARCH_BOUNDARY_V2'
class PopulationCheckpoint:
    def __init__(self,run,path):self.run=run;self.path=Path(path)
    def pending(self,ordinal):
        atomic_write_json(self.path.with_suffix('.pending.json'),dict(scope=self.run.population.scope,
            opportunity_ordinal=ordinal,state='INCOMPLETE_OPPORTUNITY_NO_AUTOMATIC_REPLAY'))
    def close_selection(self):
        atomic_write_json(self.path.with_suffix('.pending.json'),dict(scope=self.run.population.scope,
            state='FINAL_SELECTION_FROZEN_NO_SEARCH_RESUME'))
    def save(self,next_ordinal,trace):
        r=self.run;p=r.evaluation.provider;s=r.state
        stop=deepcopy(r.stop.__dict__);stop['seen_members']=sorted(stop['seen_members'])
        body=dict(schema=SCHEMA,scope=r.population.scope,next_ordinal=next_ordinal,
            pool=r.population.export(),memory=r.memory.export(),history=asdict(r.history),stop=stop,
            deployed_prompts=[v.to_dict() for v in s.prompts],initial_member_scores=list(s.initial_member_scores),
            initial_state_id=s.initial_state_id,initial_prompts=[v.to_dict() for v in p.initial_prompts],
            profiles=[dict(member=m,prompt=p.known_prompts[h].to_dict(),prompt_sha256=h,profile=profile)
                for (m,h),profile in sorted(p.member_profiles.items())],
            attempted_patterns=[list(k) for k in sorted(r.opportunities.patterns.attempted_opportunities)],
            trace=[asdict(t) for t in trace])
        atomic_write_json(self.path,{**body,'integrity_sha256':digest(body)})
        atomic_write_json(self.path.with_suffix('.pending.json'),dict(scope=r.population.scope,
            opportunity_ordinal=next_ordinal,state='BOUNDARY_COMPLETE'))
    def restore(self):
        r=self.run;p=r.evaluation.provider;s=r.state;value=read_json(self.path)
        body={k:v for k,v in value.items() if k!='integrity_sha256'}
        pending=read_json(self.path.with_suffix('.pending.json'))
        if (value.get('integrity_sha256')!=digest(body) or body.get('schema')!=SCHEMA
                or body.get('scope')!=r.population.scope or pending.get('scope')!=r.population.scope
                or pending.get('state')!='BOUNDARY_COMPLETE'
                or pending.get('opportunity_ordinal')!=body.get('next_ordinal')):
            raise SearchContractError('SEARCH_CHECKPOINT_NOT_COMPLETE_OR_SCOPE_MISMATCH')
        # Construct and verify everything in detached objects first. Cache
        # request identity and response provenance remain enforced by the scorer.
        pool=ParentPopulation(scope=r.population.scope);pool.restore(body['pool'])
        memory=PopulationEditMemory();memory.restore(body['memory'])
        known={};profiles={}
        for row in body['profiles']:
            prompt=SystemPrompt.from_dict(row['prompt']);m=row['member'];h=row['prompt_sha256']
            if (type(m) is not int or m not in range(5) or h!=prompt.prompt_hash
                    or (m,h) in profiles or len(row['profile'])!=len(s.examples)):
                raise SearchContractError('SEARCH_CHECKPOINT_PROFILE_INVALID')
            known[h]=prompt;profiles[(m,h)]=tuple(row['profile'])
            from ..benchmarks.math_response_evidence import adaptive_trajectory
            for e,raw in zip(s.examples,row['profile'],strict=True):
                s.benchmark.parse_member_output(raw,e.item)
                adaptive_trajectory(raw,member_id=m,prompt=prompt,example_id=e.item.input_id,problem=s.benchmark.format_input(e.item))
        initial=tuple(SystemPrompt.from_dict(v) for v in body['initial_prompts'])
        deployed=tuple(SystemPrompt.from_dict(v) for v in body['deployed_prompts'])
        from .binary_runtime import BinaryTeamStateStore
        def snapshot(prompts):
            scratch=BinaryTeamStateStore(benchmark=s.benchmark,examples=s.examples,prompts=prompts,
                solver=s.solver,aggregation=s.aggregation,freeze_initial_competence=True)
            try:scratch.profiles={m:profiles[(m,prompt.prompt_hash)] for m,prompt in enumerate(prompts)}
            except KeyError as error:raise SearchContractError('SEARCH_CHECKPOINT_PROFILE_MISSING') from error
            return scratch.snapshot()
        initial_state=snapshot(initial);deployed_state=snapshot(deployed)
        if (len(initial)!=5 or len(deployed)!=5 or initial_state.team_state_id!=body['initial_state_id']
                or list(initial_state.member_scores)!=body['initial_member_scores']
                or memory.competence['initial_state_id']!=initial_state.team_state_id
                or memory.competence['support_sha256']!=pool.scope['support_sha256']):
            raise SearchContractError('SEARCH_CHECKPOINT_INITIAL_FLOOR_INVALID')
        for m in range(5):
            for parent in pool.ordered(m):
                for fitness in (parent.fitness,parent.member_accuracy):
                    try:team=tuple(known[h] for h in fitness.team_prompt_hashes)
                    except KeyError as error:raise SearchContractError('SEARCH_CHECKPOINT_FITNESS_CONTEXT_MISSING') from error
                    state=snapshot(team)
                    observed=state.team_scores['vote_correct_count'] if fitness.source=='TEAM_VOTE' else state.member_scores[m]
                    if fitness.team_prompt_hashes[m]!=parent.identity or fitness.team_state_id!=state.team_state_id or fitness.correct!=observed:
                        raise SearchContractError('SEARCH_CHECKPOINT_FITNESS_OBSERVATION_MISMATCH')
        h=body['history'];history=HistoryState()
        for name in ('target_counts','failure_counts','commit_counts'):
            counts={int(k):v for k,v in h[name].items()}
            if any(k not in range(5) or type(v) is not int or v<0 for k,v in counts.items()):
                raise SearchContractError('SEARCH_CHECKPOINT_HISTORY_INVALID')
            setattr(history,name,counts)
        history.transitions=[TransitionRecord(**v) for v in h['transitions']]
        history.latest_by_member={int(k):TransitionRecord(**v) for k,v in h['latest_by_member'].items()}
        history.candidate_lineage=deepcopy(h['candidate_lineage']);history.pattern_statistics=deepcopy(h['pattern_statistics'])
        stop=body['stop']
        if (set(stop)!=set(r.stop.__dict__) or stop['no_commit_patience']!=r.stop.no_commit_patience
                or type(stop['no_commit_epochs']) is not int or stop['no_commit_epochs']<0
                or not set(stop['seen_members'])<=set(stop['eligible_members'])
                or not set(stop['eligible_members'])<=set(range(5))
                or type(body['next_ordinal']) is not int or body['next_ordinal']!=sum(history.target_counts.values())):
            raise SearchContractError('SEARCH_CHECKPOINT_STOP_HISTORY_INVALID')
        from .orchestrator import OpportunityTrace
        trace=[OpportunityTrace(**v) for v in body['trace']]
        if len(trace)!=body['next_ordinal']:raise SearchContractError('SEARCH_CHECKPOINT_TRACE_INVALID')
        # No mutations or network activity occurred before all validation passed.
        r.population.restore(pool.export());r.memory.restore(memory.export());r.history.restore(history)
        s.restore(deployed_state);s.initial_member_scores=tuple(body['initial_member_scores']);s.initial_state_id=body['initial_state_id']
        p.member_profiles=profiles;p.known_prompts=known;p.initial_prompts=initial
        r.stop.__dict__.update(deepcopy(stop));r.stop.seen_members=set(stop['seen_members']);r.stop.eligible_members=tuple(stop['eligible_members'])
        r.opportunities.patterns.attempted_opportunities={tuple(v) for v in body['attempted_patterns']}
        r.resume_ordinal=body['next_ordinal'];r.resume_trace=trace
        return dict(state='BOUNDARY_RESTORED',next_ordinal=body['next_ordinal'],real_api_authorized=False)
