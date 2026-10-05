"""Bounded Optimize-only search; no official GEPA survival or deployment authority."""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from ... import versions
from ...evaluation.semantic_mutable_contract import candidate_failed_checks, mutation_shape, normalized_procedure
from ...local_optimizers.schemas import LocalEvidenceExample, LocalOptimizationTask, LocalOptimizerBudget
from ...persistence.durable_io import append_jsonl
from ..context import SearchContextComposer
from ..schemas import SearchCandidate, SearchResult, SearchContractError


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest()


@dataclass(frozen=True)
class Layer1Config:
    identity_version: str = versions.LAYER1_RESPONSIBILITY_SEARCH_VERSION
    metric_limit: int = 36
    panel_size: int = 6
    max_generations: int = 6
    k_local_return: int = 4
    max_prompt_chars: int = 3000
    candidate_contract: str = versions.SEMANTIC_MUTABLE_CONTRACT_VERSION
    official_gepa_fidelity: bool = False

    def __post_init__(self):
        if asdict(self)!=dict(identity_version=versions.LAYER1_RESPONSIBILITY_SEARCH_VERSION,
            metric_limit=36,panel_size=6,max_generations=6,k_local_return=4,max_prompt_chars=3000,
            candidate_contract=versions.SEMANTIC_MUTABLE_CONTRACT_VERSION,official_gepa_fidelity=False):
            raise SearchContractError('LAYER1_FROZEN_CONFIG_MISMATCH')

    def identity(self):return digest(asdict(self))


PROPOSER_INSTRUCTION = '''Edit one generic mathematical decision procedure for the selected member.
Use only this Optimize evidence and its single supplied repair objective. Preservation/boundary
examples constrain collateral risk; do not add another focused repair objective for a boundary.
Diagnose observed failures and propose a distinct, complete reusable reasoning procedure.
Mathematical representation guidance, exactness, domain/unit checks, substitution and verification
are allowed. A changed append-only procedure is allowed. Do not include supplied example facts,
entities, answers, or memorized cases. Do not control external response presentation: parser
markers, line placement/count, response-only wording, Markdown/fences, JSON/XML Solver schemas,
or commentary/confidence output. Solver formatting belongs to the immutable benchmark shell.
Return a single JSON object with exactly one key "decision_procedure" containing the complete
procedure as a string of at most 3000 characters. This JSON is the optimizer envelope only;
the procedure itself must contain reasoning guidance. Do not return an unchanged parent.'''


def evaluation_panel(task, size=6):
    rows=tuple({r.example_id:r for r in (*task.search_examples,*task.local_validation_examples)}.values())
    repair=[r for r in rows if 'repair' in r.tags or 'focus_repair_v2' in r.tags]
    boundary=[r for r in rows if r not in repair]
    ordered=[]
    for i in range(max(len(repair),len(boundary))):
        if i<len(repair):ordered.append(repair[i])
        if i<len(boundary):ordered.append(boundary[i])
    return tuple(ordered[:size])


def observation_record(row,observation):
    return dict(example_id=row.example_id,problem=row.input_payload,reference=row.gold,
        prediction=observation.parsed_answer,correct=bool(observation.valid and observation.correct),
        valid=observation.valid,invalid_reason=observation.failure_reason,
        role_and_lane=list(row.tags),evaluator_feedback=row.textual_feedback,
        visible_reasoning_required=False)


def generation_input(task,parent,root_records,recent_records,archive,generation):
    return PROPOSER_INSTRUCTION+'\n'+json.dumps(dict(
        schema='LAYER1_SOLVER_OBSERVATION_FEEDBACK_V1',target_member=task.target_member,
        opportunity=task.task_id,search_context=task.optimization_context,
        current_parent=parent,root_observations=root_records,
        recent_candidate_observations=recent_records,candidate_archive=archive,
        generation=generation),sort_keys=True,separators=(',',':'),ensure_ascii=True)


class ResponsibilityConditionedOptimizer:
    def __init__(self, *, evaluator, reflection_lm, accounting_reader, run_root, config=None):
        self.evaluator=evaluator;self.reflection_lm=reflection_lm
        self.accounting_reader=accounting_reader;self.run_root=Path(run_root)
        self.config=config or Layer1Config()

    async def search_task(self,task,opportunity_id):
        if (task.backend_state is not None or task.target_member not in range(5)
                or task.budget.max_metric_calls!=self.config.metric_limit
                or task.budget.max_returned_candidates!=self.config.k_local_return):
            raise SearchContractError('LAYER1_TASK_CONTRACT_MISMATCH')
        if (self.evaluator.solver_contract_id!=task.solver_contract_id or
                self.evaluator.output_contract_id!=task.output_contract_id):
            raise SearchContractError('LAYER1_SOLVER_PORT_MISMATCH')
        panel=evaluation_panel(task,self.config.panel_size)
        if not panel:raise SearchContractError('LAYER1_NO_LEGAL_PANEL')
        lineage=self.run_root/(task.task_id+'.lineage.jsonl')
        if lineage.exists():raise SearchContractError('LAYER1_FRESH_STATE_REQUIRED')
        all_examples=(*task.search_examples,*task.local_validation_examples)
        if candidate_failed_checks(task.parent_prompt,parent_prompt='__root_contract_check__',examples=all_examples):
            raise SearchContractError('LAYER1_PARENT_CONTRACT_VIOLATION')
        before=dict(self.accounting_reader());metric=solver_calls=solver_tokens=0
        events=[];pool=[];seen={normalized_procedure(task.parent_prompt)};recent=[]
        def evaluate(prompt):
            nonlocal metric,solver_calls,solver_tokens
            if metric+len(panel)>task.budget.max_metric_calls:
                raise SearchContractError('LAYER1_METRIC_LIMIT_PRE_SOLVER')
            metric+=len(panel)
            obs=[self.evaluator.evaluate(prompt,row) for row in panel]
            solver_calls+=sum(o.provider_called for o in obs)
            solver_tokens+=sum(o.input_tokens+o.output_tokens for o in obs)
            return [observation_record(row,o) for row,o in zip(panel,obs,strict=True)]
        root_records=evaluate(task.parent_prompt)
        root_bits=[int(r['correct']) for r in root_records];root_score=sum(root_bits)
        selected_parent=task.parent_prompt;best_score=root_score;best_bits=root_bits
        parent_hash=hashlib.sha256(task.parent_prompt.encode()).hexdigest()
        stop='LAYER1_GENERATION_BOUND_REACHED'
        for generation in range(1,self.config.max_generations+1):
            if metric+len(panel)>task.budget.max_metric_calls:
                stop='LAYER1_METRIC_BOUND_REACHED';break
            prompt=generation_input(task,selected_parent,root_records,recent,events,generation)
            packet_hash=hashlib.sha256(prompt.encode()).hexdigest()
            raw=self.reflection_lm(prompt)
            try:
                obj=json.loads(raw)
                if not isinstance(obj,dict) or set(obj)!={'decision_procedure'} or not isinstance(obj['decision_procedure'],str):
                    raise ValueError('envelope')
                proposed=obj['decision_procedure']
            except (ValueError,TypeError):
                row=dict(generation=generation,status='CONTRACT_INVALID',failed_checks=['invalid_structure'],
                    evidence_packet_hash=packet_hash,source_proposal_call=generation)
                events.append(row);append_jsonl(lineage,row);recent=[dict(contract_feedback=row)];continue
            h=hashlib.sha256(proposed.encode()).hexdigest()
            checks=candidate_failed_checks(proposed,parent_prompt=task.parent_prompt,
                examples=all_examples,max_chars=self.config.max_prompt_chars)
            norm=normalized_procedure(proposed)
            if norm in seen and not checks:checks=('duplicate_proposal',)
            seen.add(norm)
            current_parent_hash=hashlib.sha256(selected_parent.encode()).hexdigest()
            common=dict(candidate_id=f'layer1:{generation}:{h[:12]}',prompt_hash=h,parent_hash=current_parent_hash,
                root_parent_hash=parent_hash,target_member=task.target_member,generation=generation,
                source_proposal_call=generation,evidence_packet_hash=packet_hash,
                mutation_shape=mutation_shape(proposed,task.parent_prompt))
            if checks:
                row=dict(common,status='CONTRACT_INVALID',failed_checks=list(checks),solver_evaluated=False)
                events.append(row);append_jsonl(lineage,row);recent=[dict(contract_feedback=row)];continue
            observed=evaluate(proposed);bits=[int(r['correct']) for r in observed]
            fixed=sum(not a and b for a,b in zip(root_bits,bits,strict=True))
            broken=sum(a and not b for a,b in zip(root_bits,bits,strict=True))
            score=sum(bits);delta=score-root_score;parent_delta=score-best_score;positive=parent_delta>0
            details=dict(common,local_examples_evaluated=[r.example_id for r in panel],
                local_correct_count=score,local_correct_delta=delta,local_newly_fixed=fixed,
                local_root_correct_count=root_score,local_parent_correct_count=best_score,
                local_parent_correct_delta=parent_delta,
                local_parent_newly_fixed=sum(not a and b for a,b in zip(best_bits,bits,strict=True)),
                local_parent_newly_broken=sum(a and not b for a,b in zip(best_bits,bits,strict=True)),
                local_newly_broken=broken,local_invalid_count=sum(not r['valid'] for r in observed),
                local_search_status='POSITIVE' if positive else 'NEGATIVE' if parent_delta<0 else 'NEUTRAL',
                locally_positive=positive,locally_rejected=not positive,local_acceptance_delta=parent_delta,
                local_effect_scope='fixed_optimize_panel',local_preservation_loss=broken,
                changed=True,contract_valid=True,solver_evaluated=True,duplicate=False,
                opportunity_id=opportunity_id,official_gepa_frontier=False,
                export_reason='eligible_all_admissible_scored_pool',exported=False)
            candidate=SearchCandidate(common['candidate_id'],proposed,float(score),{},details)
            pool.append(candidate)
            row=dict(details,status='LOCALLY_EVALUATED');events.append(row);append_jsonl(lineage,row)
            recent=observed
            if score>best_score:selected_parent=proposed;best_score=score;best_bits=bits
        ranked=sorted(pool,key=lambda c:(-c.backend_details['local_newly_fixed'],
            c.backend_details['local_newly_broken'],-c.backend_details['local_correct_count'],
            c.backend_details['generation']))
        chosen=ranked[:self.config.k_local_return]
        candidates=tuple(SearchCandidate(c.candidate_id,c.prompt,c.search_score,c.lineage,
            {**c.backend_details,'exported':True,'export_reason':'bounded_rank_after_all_scored_pool',
             'LOCAL_REJECTED_EXPORTED':c.backend_details['locally_rejected']}) for c in chosen)
        exported={c.candidate_id for c in candidates}
        outcomes=[{**r,'exported':r.get('candidate_id') in exported} for r in events]
        after=dict(self.accounting_reader())
        meta_tokens=sum(after.get(k,0)-before.get(k,0) for k in ('input_tokens','output_tokens'))
        meta_calls=after.get('successful_calls',0)-before.get('successful_calls',0)
        telemetry=dict(proposal_count=len(events),physical_proposal_generations=meta_calls,
            unique_candidates=sum(r.get('solver_evaluated',False) for r in events),
            contract_rejects=sum(r['status']=='CONTRACT_INVALID' for r in events),
            local_solver_reached=len(pool),local_positive=sum(c.backend_details['locally_positive'] for c in pool),
            local_negative=sum(c.backend_details['local_parent_correct_delta']<0 for c in pool),
            local_neutral=sum(c.backend_details['local_parent_correct_delta']==0 for c in pool),
            team_candidate_count=len(candidates),strict_rejected_exported_count=sum(c.backend_details['locally_rejected'] for c in candidates),
            local_metric_evaluations=metric,metric_limit=task.budget.max_metric_calls)
        state=dict(protocol_hash=self.config.identity(),backend=versions.LAYER1_RESPONSIBILITY_SEARCH_VERSION,
            official_gepa_fidelity=False,telemetry=telemetry,proposal_diagnostics={'proposal_outcomes':outcomes},
            callback_events=outcomes,operational_failure=False,
            token_accounting={'solver_tokens':solver_tokens,'search_meta_tokens':meta_tokens})
        return SearchResult(candidates,stop,state,solver_calls,meta_calls,solver_tokens,meta_tokens,
            len(events),len(candidates),0,0,telemetry['strict_rejected_exported_count'])


class ResponsibilityConditionedEngine:
    identity=versions.LAYER1_RESPONSIBILITY_SEARCH_VERSION

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
