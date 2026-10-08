"""Generic bounded local task, observation and search primitives."""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from .. import current_contract as versions
from ..local_optimizers.schemas import LocalOptimizationTask, LocalEvidenceExample, LocalOptimizerBudget
from ..evaluation.semantic_mutable_contract import candidate_failed_checks, mutation_shape, normalized_procedure
from ..persistence.durable_io import append_jsonl
from .private_action_memory import edit_action
from .schemas import SearchCandidate, SearchResult, SearchContractError
from .current_context import SearchContextComposer

@dataclass(frozen=True)
class MemoryLocalTask(LocalOptimizationTask):
    anchor_example: LocalEvidenceExample | None = None
    responsibility_lane: str = 'general'

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()


def observation_record(row,observation):
    record = dict(example_id=row.example_id,problem=row.input_payload,reference=row.gold,
        prediction=observation.parsed_answer,correct=bool(observation.valid and observation.correct),
        valid=observation.valid,invalid_reason=observation.failure_reason,
        role_and_lane=list(row.tags),evaluator_feedback=row.textual_feedback,
        visible_reasoning_required=False)
    if getattr(observation, 'solver_trajectory', None) is not None:
        from ..benchmarks.math_visible_trajectory import validate_adaptive_trajectory
        record['solver_trajectory'] = validate_adaptive_trajectory(observation.solver_trajectory,
            example_id=row.example_id, problem=row.input_payload)
    return record


def build_optimizer_context(task,parent,observations,memory):
    if len(json.dumps(memory,sort_keys=True,separators=(',',':'),ensure_ascii=True))>1200:
        raise SearchContractError('MEMORY_CONTEXT_LIMIT_EXCEEDED')
    # No example identifiers, candidate archive, generation or controller metadata.
    current=[{k:r[k] for k in ('problem','reference','prediction','correct','valid','invalid_reason','role_and_lane','evaluator_feedback')}
             for r in observations]
    if not observations or any('solver_trajectory' not in r for r in observations):
        raise SearchContractError('LAYER1_VISIBLE_TRAJECTORY_MISSING')
    trajectory_enabled = True
    if trajectory_enabled:
        from ..benchmarks.math_visible_trajectory import validate_adaptive_trajectory
        for row, value in zip(observations, current, strict=True):
            if 'solver_trajectory' not in row:
                raise SearchContractError('LAYER1_VISIBLE_TRAJECTORY_MISSING')
            value['solver_trajectory'] = validate_adaptive_trajectory(row['solver_trajectory'],
                example_id=row['example_id'], member_id=task.target_member, prompt=parent,
                problem=row['problem'])
    from .optimization_evidence import INPUT
    return dict(schema=INPUT,
        current_parent=parent,repair_objective=task.responsibility_lane+' failures',
        current_panel_observations=current,retrieved_memory=memory)

class BoundedMemoryOptimizer:
    def __init__(self, *, evaluator, reflection_lm, accounting_reader, run_root):
        self.evaluator=evaluator;self.reflection_lm=reflection_lm
        self.accounting_reader=accounting_reader;self.run_root=Path(run_root)
        self.config=self.config_factory();self.memory=None
        self.observation_observer=None

    async def search_task(self,task,opportunity_id):
        from .optimization_evidence import INPUT
        if self.config.optimizer_input_schema!=INPUT:
            raise SearchContractError('CURRENT_LAYER1_EVIDENCE_REQUIRED')
        return await self._search_evidence_task(task,opportunity_id)

    async def _search_evidence_task(self,task,opportunity_id):
        from .optimization_evidence import (INPUT, coverage_effect, actual_diff,
            prompt_id, executability_checks)
        if (task.backend_state is not None or task.target_member not in range(5)
                or task.budget.max_metric_calls!=42 or task.budget.max_returned_candidates!=4
                or self.memory is None or self.evaluator.solver_contract_id!=task.solver_contract_id
                or self.evaluator.output_contract_id!=task.output_contract_id):
            raise SearchContractError('EVIDENCE_LAYER1_TASK_MISMATCH')
        mutation=tuple(task.search_examples);validation=tuple(task.local_validation_examples)
        if len(mutation)!=3 or len(validation)!=3 or {r.example_id for r in mutation}&{r.example_id for r in validation}:
            raise SearchContractError('OPTIMIZE_ROLE_OVERLAP_OR_SIZE')
        path=self.run_root/(task.task_id+'.lineage.jsonl')
        if path.exists():raise SearchContractError('LAYER1_FRESH_STATE_REQUIRED')
        pattern=json.loads(task.selected_pattern)
        if pattern['pattern_id'] is None:
            return SearchResult((),'NO_ACTIONABLE_GRADIENT',{'operational_failure':False,
                'telemetry':{'proposal_count':0,'local_metric_evaluations':0}},0,0,0,0,0,0,0,0,0)
        examples=(*mutation,*validation)
        if candidate_failed_checks(task.parent_prompt,parent_prompt='__root_contract_check__',examples=examples):
            raise SearchContractError('LAYER1_PARENT_CONTRACT_VIOLATION')
        before=dict(self.accounting_reader());metric=solver_calls=solver_tokens=0
        events=[];pool=[];seen={normalized_procedure(task.parent_prompt)}
        def evaluate(prompt,rows):
            nonlocal metric,solver_calls,solver_tokens
            if metric+len(rows)>42:raise SearchContractError('LAYER1_METRIC_LIMIT_PRE_SOLVER')
            metric+=len(rows)
            observations=[self.evaluator.evaluate(prompt,row) for row in rows]
            from ..benchmarks.math_visible_trajectory import validate_adaptive_trajectory
            for row,obs in zip(rows,observations,strict=True):
                if getattr(obs,'solver_trajectory',None) is None:
                    raise SearchContractError('LAYER1_VISIBLE_TRAJECTORY_MISSING')
                validate_adaptive_trajectory(obs.solver_trajectory,example_id=row.example_id,
                    member_id=task.target_member,prompt=prompt,problem=row.input_payload)
            solver_calls+=sum(o.provider_called for o in observations)
            solver_tokens+=sum(o.input_tokens+o.output_tokens for o in observations)
            return [dict(observation_record(row,obs),evidence_revision=INPUT)
                for row,obs in zip(rows,observations,strict=True)]
        def mapping(records):return {r['example_id']:{k:r[k] for k in ('correct','valid')} for r in records}
        root_mut=evaluate(task.parent_prompt,mutation);root_val=evaluate(task.parent_prompt,validation)
        selected_parent=task.parent_prompt;current_mut=root_mut;current_val=root_val
        root_score=sum(r['correct'] for r in root_val)
        def journal(row):
            events.append(row);append_jsonl(path,row)
            if self.observation_observer:
                self.observation_observer('GENERATION_COMPLETE',dict(opportunity_id=opportunity_id,
                    target_member=task.target_member,generation=row['generation'],row=row))
        for generation in range(1,7):
            memory=self.memory.read_for_member(task.target_member,task.responsibility_lane,
                pattern_id=pattern['pattern_id'],hypothesis=pattern['generalized_gradient'])
            packet=self.prompt_builder(task,selected_parent,current_mut,memory)
            obj=None
            try:obj=json.loads(self.reflection_lm(packet))
            except (ValueError,TypeError):pass
            common=dict(generation=generation,opportunity_id=opportunity_id,target_member=task.target_member,
                evidence_packet_hash=prompt_id(packet),parent_hash=prompt_id(selected_parent),
                root_parent_hash=prompt_id(task.parent_prompt))
            if obj=={'decision':'NO_SAFE_EDIT'}:
                journal(dict(common,status='NO_SAFE_EDIT',solver_evaluated=False));continue
            if (not isinstance(obj,dict) or set(obj)!={'decision','decision_procedure','change_summary'}
                    or obj.get('decision')!='PROPOSE_EDIT' or not isinstance(obj.get('decision_procedure'),str)
                    or not isinstance(obj.get('change_summary'),str) or not 0<len(obj['change_summary'])<=240):
                journal(dict(common,status='CONTRACT_INVALID',failed_checks=['invalid_structure'],solver_evaluated=False));continue
            proposed=obj['decision_procedure'];h=prompt_id(proposed)
            checks=candidate_failed_checks(proposed,parent_prompt=selected_parent,examples=examples,max_chars=3000)
            checks=(*checks,*executability_checks(proposed,private_texts=(*task.private_feedback_texts,
                *(r['solver_trajectory']['visible_solution'] for r in current_mut))))
            norm=normalized_procedure(proposed)
            if norm in seen:checks=(*checks,'duplicate_proposal')
            seen.add(norm);cid=f'layer1:{generation}:{h[:12]}'
            common.update(candidate_id=cid,prompt_hash=h)
            if checks:
                journal(dict(common,status='CONTRACT_INVALID',failed_checks=sorted(set(checks)),solver_evaluated=False));continue
            # Generation is complete before either candidate evaluation occurs.
            child_mut=evaluate(proposed,mutation);child_val=evaluate(proposed,validation)
            mut=coverage_effect(mapping(root_mut),mapping(child_mut),scope='mutation_root')
            val=coverage_effect(mapping(root_val),mapping(child_val),scope='search_validation_root')
            immediate=coverage_effect(mapping(current_val),mapping(child_val),scope='search_validation_actual_parent')
            lineage=dict(candidate_id=cid,opportunity_id=opportunity_id,member=task.target_member,
                parent_prompt_id=prompt_id(selected_parent),child_prompt_id=h,
                full_parent_prompt_id=prompt_id(task.parent_prompt),pattern_id=pattern['pattern_id'],
                supporting_example_ids=list(task.pattern_support),repair_hypothesis=pattern['generalized_gradient'],
                intended_edit=obj['change_summary'],actual_diff=actual_diff(selected_parent,proposed),
                action=edit_action(selected_parent,proposed),attribution='compound_edit_no_clause_causal_claim',
                expected_behavior=pattern['generalized_gradient'],
                effects=dict(mutation=mut,search_validation=val,actual_parent_validation=immediate),
                status='LOCALLY_SUPPORTED' if val['member_delta']>0 and not val['broken_ids'] else 'INCONCLUSIVE',
                status_history=['PROPOSED'],provenance=dict(split='optimize',generation=generation,
                    evidence_packet_sha256=common['evidence_packet_hash'],solver_interface='MATH_SOLVER_INTERFACE_V6'))
            lineage['status_history'].append(lineage['status'])
            lineage['provenance']['local_request_identities']={name:{r['example_id']:
                r['solver_trajectory']['source']['request_sha256'] for r in rows}
                for name,rows in dict(root_mutation=root_mut,root_validation=root_val,
                    actual_parent_validation=current_val,child_mutation=child_mut,child_validation=child_val).items()}
            details=dict(common,edit_lineage=lineage,changed=True,contract_valid=True,solver_evaluated=True,
                duplicate=False,local_correct_count=sum(r['correct'] for r in child_val),
                local_correct_delta=val['member_delta'],local_newly_fixed=len(val['fixed_ids']),
                local_newly_broken=len(val['broken_ids']),local_parent_correct_delta=immediate['member_delta'],
                local_parent_newly_fixed=len(immediate['fixed_ids']),local_parent_newly_broken=len(immediate['broken_ids']),
                local_invalid_count=len(val['invalid_ids']),locally_positive=val['member_delta']>0,
                locally_rejected=val['member_delta']<=0,local_effect_scope='independent_optimize_search_validation',
                preservation_locally_measurable=any(r['correct'] for r in root_val),
                memory_action=lineage['action'],exported=False)
            self.memory.observe_edit(lineage)
            candidate=SearchCandidate(cid,proposed,float(details['local_correct_count']),{},details)
            pool.append(candidate);journal(dict(details,status='LOCALLY_EVALUATED'))
            # SearchValidation updates the next parent; never the pre-generation packet.
            if immediate['member_delta']>0 and not immediate['broken_ids']:
                selected_parent=proposed;current_mut=child_mut;current_val=child_val
        def rank(c):
            e=c.backend_details['edit_lineage']['effects']
            return (-e['search_validation']['member_delta'],len(e['search_validation']['broken_ids']),
                -e['mutation']['member_delta'],c.backend_details['generation'])
        chosen=sorted(pool,key=rank)[:4]
        candidates=tuple(SearchCandidate(c.candidate_id,c.prompt,c.search_score,c.lineage,
            {**c.backend_details,'exported':True,'export_reason':'validation_rank_all_admissible_no_local_gate'}) for c in chosen)
        exported={c.candidate_id for c in candidates}
        after=dict(self.accounting_reader())
        calls=after.get('successful_calls',0)-before.get('successful_calls',0)
        tokens=sum(after.get(k,0)-before.get(k,0) for k in ('input_tokens','output_tokens'))
        telemetry=dict(proposal_count=len(events),local_metric_evaluations=metric,metric_limit=42,
            mutation_size=3,search_validation_size=3,local_solver_reached=len(pool),
            contract_rejects=sum(r['status']=='CONTRACT_INVALID' for r in events),
            no_safe_edits=sum(r['status']=='NO_SAFE_EDIT' for r in events),root_correct_count=root_score,
            team_candidate_count=len(candidates),local_positive=sum(c.backend_details['locally_positive'] for c in pool),
            memory=self.memory.audit())
        state=dict(protocol_hash=self.config.identity(),backend=self.config.identity_version,telemetry=telemetry,
            operational_failure=False,proposal_diagnostics={'proposal_outcomes':[{**r,'exported':r.get('candidate_id') in exported} for r in events]},
            token_accounting={'solver_tokens':solver_tokens,'search_meta_tokens':tokens})
        return SearchResult(candidates,'LAYER1_GENERATION_BOUND_REACHED',state,solver_calls,calls,solver_tokens,tokens,
            len(events),len(candidates),0,0,sum(c.backend_details['locally_rejected'] for c in candidates))

class LocalTaskEngine:
    identity='INDEPENDENT_OPTIMIZE_VALIDATION_SEARCH_V1'
    def __init__(self,optimizer,seed):self.optimizer=optimizer;self.seed=seed

    def make_task(self,opportunity,context):
        rows=(*opportunity.evidence.mutation_evidence,*opportunity.evidence.search_validation_evidence)
        if any(r.source_split!='optimize' for r in rows):raise SearchContractError('LAYER1_HELDOUT_ACCESS')
        signal=opportunity.diagnosis.responsibility[opportunity.target_member]
        objective=dict(target_member=opportunity.target_member,objective=dict(opportunity.objective),
            responsibility=asdict(signal),seed=self.seed,
            evidence_role_memberships={xid:sorted(set().union(*(r.roles for r in rows if r.example_id==xid))) for xid in sorted({r.example_id for r in rows})})
        base=json.dumps(objective,sort_keys=True,separators=(',',':'))
        actual_context=SearchContextComposer().compose(base,context.pattern_view,context.memory_view)
        def local(row):
            s=row.signals
            return LocalEvidenceExample(row.example_id,s['input_payload'],s['gold'],s.get('target_output'),
                s.get('feedback'),tuple(s.get('legacy_tags',()))+tuple(sorted(row.roles-{'MUTATION','SEARCH_VALIDATION','TEAM_PROBE'}))+('lane='+str(s.get('lane','general')),))
        return LocalOptimizationTask('layer1_'+hashlib.sha256(opportunity.opportunity_id.encode()).hexdigest(),
            opportunity.parent_prompt,tuple(map(local,opportunity.evidence.mutation_evidence)),
            tuple(map(local,opportunity.evidence.search_validation_evidence)),actual_context,
            self.optimizer.evaluator.solver_contract_id,self.optimizer.evaluator.output_contract_id,self.seed,
            LocalOptimizerBudget(opportunity.search_budget['metric_calls'],3,self.optimizer.config.k_local_return),
            target_member=opportunity.target_member)

    async def search(self,opportunity,context):
        if hasattr(self.optimizer.evaluator,'observe_member'):
            self.optimizer.evaluator.observe_member(opportunity.target_member)
        return await self.optimizer.search_task(self.make_task(opportunity,context),opportunity.opportunity_id)
